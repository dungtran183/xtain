from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from crosstaint.pseudonym.model import HGTResolver, HeteroGraph, HeteroNode
from crosstaint.types import EdgeType, NodeFeatures


@dataclass
class PseudonymGraphDataset:
    pairs: list[tuple[str, str, float]]
    node_features: dict[str, NodeFeatures]
    graph: HeteroGraph | None = None

    def __post_init__(self) -> None:
        self._build_graph()

    def _build_graph(self) -> None:
        node_map: dict[str, tuple[str, str]] = {}
        for node_id, nf in self.node_features.items():
            chain = getattr(nf, "chain", "ethereum")
            node_map[node_id] = (chain, node_id)

        chain_nodes: dict[str, list[HeteroNode]] = {}
        for node_id, (chain, _) in node_map.items():
            nf = self.node_features[node_id]
            vec = nf.to_vector()
            node = HeteroNode(
                node_id=node_id,
                chain=chain,
                features=torch.tensor(vec, dtype=torch.float32),
            )
            chain_nodes.setdefault(chain, []).append(node)

        edges_by_type: dict[str, list[tuple[str, str]]] = {
            et: [] for et in HGTResolver.EDGE_TYPES
        }

        seen_pairs: set[tuple[str, str]] = set()
        for a, b, _ in self.pairs:
            if (a, b) in seen_pairs or (b, a) in seen_pairs:
                continue
            seen_pairs.add((a, b))

            if a not in node_map or b not in node_map:
                continue

            chain_a, _ = node_map[a]
            chain_b, _ = node_map[b]

            edge_type = (
                EdgeType.INTRA_CHAIN_TRANSFER
                if chain_a == chain_b
                else EdgeType.CROSS_CHAIN_BRIDGE
            )
            edges_by_type.setdefault(edge_type, []).append((a, b))

        self.graph = HeteroGraph(nodes=chain_nodes, edges=edges_by_type)


@dataclass
class PseudonymTrainer:
    resolver: HGTResolver
    train_pairs: list[tuple[str, str, float]]
    val_pairs: list[tuple[str, str, float]]
    node_features: dict[str, NodeFeatures]
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4

    _best_state: Optional[dict] = field(default=None, init=False, repr=False)
    _history: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolver = self.resolver.to(self.device)

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.resolver.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_dataset = PseudonymGraphDataset(
            self.train_pairs, self.node_features
        )
        self.val_dataset = PseudonymGraphDataset(
            self.val_pairs, self.node_features
        )

    def _pair_labels_to_tensor(
        self, pairs: list[tuple[str, str, float]]
    ) -> torch.Tensor:
        labels = [label for _, _, label in pairs]
        return torch.tensor(labels, dtype=torch.float32, device=self.device)

    def train(self) -> dict:
        self._history = {"loss": [], "val_auc": []}
        best_val_auc = 0.0
        self._best_state = None

        for epoch in range(self.epochs):
            self.resolver.train()
            self.optimizer.zero_grad()

            train_emb = self.resolver(self.train_dataset.graph)
            train_labels = self._pair_labels_to_tensor(self.train_pairs)

            train_pair_ids = [(a, b) for a, b, _ in self.train_pairs]
            logits = self.resolver.pair_logits(train_pair_ids, train_emb, self.train_dataset.graph)
            loss = self.criterion(logits, train_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.resolver.parameters(), 1.0)
            self.optimizer.step()

            self._history["loss"].append(loss.item())

            val_metrics = self._eval_epoch()
            val_auc = val_metrics.get("auc", 0.0)
            self._history["val_auc"].append(val_auc)

            self.scheduler.step(loss.item())

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self._best_state = {
                    k: v.cpu().clone()
                    for k, v in self.resolver.state_dict().items()
                }

        if self._best_state is not None:
            self.resolver.load_state_dict(self._best_state)

        return self._history

    def _eval_epoch(self) -> dict:
        self.resolver.eval()
        with torch.no_grad():
            val_emb = self.resolver(self.val_dataset.graph)
            val_labels = self._pair_labels_to_tensor(self.val_pairs)
            pair_ids = [(a, b) for a, b, _ in self.val_pairs]
            scores = self.resolver.pair_scores(pair_ids, val_emb, self.val_dataset.graph)
        return _binary_metrics(scores, val_labels)

    def evaluate(self, test_pairs: list[tuple[str, str, float]]) -> dict:
        test_dataset = PseudonymGraphDataset(test_pairs, self.node_features)
        self.resolver.eval()

        with torch.no_grad():
            test_emb = self.resolver(test_dataset.graph)
            test_labels = self._pair_labels_to_tensor(test_pairs)
            pair_ids = [(a, b) for a, b, _ in test_pairs]
            scores = self.resolver.pair_scores(pair_ids, test_emb, test_dataset.graph)
        return _binary_metrics(scores, test_labels)


def _binary_metrics(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if scores.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.5}
    predictions = (scores >= 0.5).float()
    positives = (labels >= 0.5).float()
    negatives = 1.0 - positives
    tp = (predictions * positives).sum().item()
    fp = (predictions * negatives).sum().item()
    fn = ((1.0 - predictions) * positives).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": _rank_auc(scores, labels),
    }


def _rank_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = (labels >= 0.5).float()
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32, device=scores.device)
    rank_sum_pos = (ranks * labels).sum().item()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))
