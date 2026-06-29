from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch

from crosstaint.types import (
    EdgeType,
    SimilarityOutput,
)


@dataclass
class HeteroNode:
    node_id: str
    chain: str
    features: torch.Tensor


@dataclass
class HeteroGraph:
    nodes: dict[str, list[HeteroNode]]
    edges: dict[str, list[tuple[str, str]]]

    def node_ids(self, chain: str) -> list[str]:
        if chain == "global":
            return [
                node.node_id
                for chain_nodes in self.nodes.values()
                for node in chain_nodes
            ]
        return [n.node_id for n in self.nodes.get(chain, [])]

    def edge_pairs(self, edge_type: str) -> list[tuple[str, str]]:
        return self.edges.get(edge_type, [])


class HGTAttention(nn.Module):
    RELATION_TYPES = (
        EdgeType.INTRA_CHAIN_TRANSFER,
        EdgeType.CROSS_CHAIN_BRIDGE,
        EdgeType.DEX_SWAP,
    )

    def __init__(self, hidden_dim: int = 256, n_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.relation_proj = nn.Parameter(
            torch.empty(len(self.RELATION_TYPES), n_heads, self.head_dim, self.head_dim)
        )
        nn.init.xavier_uniform_(self.relation_proj)
        self.relation_index = {name: idx for idx, name in enumerate(self.RELATION_TYPES)}

        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        source_emb: torch.Tensor,
        target_emb: torch.Tensor,
        rel_type: str,
    ) -> torch.Tensor:
        batch_size = source_emb.size(0)

        q = self.q_proj(source_emb).view(batch_size, self.n_heads, self.head_dim)
        k = self.k_proj(target_emb).view(batch_size, self.n_heads, self.head_dim)
        v = self.v_proj(target_emb).view(batch_size, self.n_heads, self.head_dim)
        relation_id = self.relation_index.get(rel_type, 0)
        rel = self.relation_proj[relation_id]
        k = torch.einsum("bhd,hde->bhe", k, rel)

        attn_scores = (q * k).sum(dim=-1) / self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights.unsqueeze(-1) * v).view(batch_size, self.hidden_dim)
        return self.out_proj(out)


class HGTLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.attention = HGTAttention(hidden_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        node_embeddings: dict[str, torch.Tensor],
        edge_index_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        updated: dict[str, torch.Tensor] = {}

        for chain, emb in node_embeddings.items():
            n_nodes = emb.size(0)
            aggregated = torch.zeros_like(emb)
            edge_count = torch.zeros(n_nodes, device=emb.device)

            for edge_type, (src_idx, dst_idx) in edge_index_by_type.items():
                if src_idx.numel() == 0:
                    continue

                src_emb = emb[src_idx]
                dst_emb = emb[dst_idx]

                attended = self.attention(src_emb, dst_emb, edge_type)

                edge_count.scatter_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float))
                aggregated.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(attended), attended)

            aggregated = aggregated / edge_count.clamp(min=1.0).unsqueeze(-1)

            aggregated = self.norm1(emb + aggregated)
            aggregated = self.norm2(aggregated + self.ffn(aggregated))
            updated[chain] = aggregated

        return updated


class HGTResolver(nn.Module):
    EDGE_TYPES = {
        EdgeType.INTRA_CHAIN_TRANSFER,
        EdgeType.CROSS_CHAIN_BRIDGE,
        EdgeType.DEX_SWAP,
    }

    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        feature_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HGTLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.sim_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph: HeteroGraph) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        flat_nodes = [
            node
            for chain_nodes in graph.nodes.values()
            for node in chain_nodes
        ]
        node_index = {node.node_id: idx for idx, node in enumerate(flat_nodes)}

        if not flat_nodes:
            return {"global": torch.empty((0, self.hidden_dim), device=device)}

        edge_index_by_type: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            et: (torch.tensor([], dtype=torch.long, device=device),
                 torch.tensor([], dtype=torch.long, device=device))
            for et in self.EDGE_TYPES
        }

        features_list = []
        for node in flat_nodes:
            if isinstance(node.features, torch.Tensor):
                features_list.append(node.features)
            else:
                features_list.append(torch.tensor(node.features, dtype=torch.float32))

        features = torch.stack(features_list, dim=0).to(device)
        node_embeddings: dict[str, torch.Tensor] = {"global": self.input_proj(features)}

        for edge_type, pairs in graph.edges.items():
            if edge_type not in self.EDGE_TYPES:
                continue
            valid_pairs = [
                (src, dst)
                for src, dst in pairs
                if src in node_index and dst in node_index
            ]
            if not valid_pairs:
                continue
            src_idx = torch.tensor(
                [node_index[src] for src, _ in valid_pairs],
                dtype=torch.long,
                device=device,
            )
            dst_idx = torch.tensor(
                [node_index[dst] for _, dst in valid_pairs],
                dtype=torch.long,
                device=device,
            )
            edge_index_by_type[edge_type] = (src_idx, dst_idx)

        emb = node_embeddings
        for layer in self.layers:
            emb = layer(emb, edge_index_by_type)

        return emb

    def _embedding_by_node(
        self,
        embeddings: dict[str, torch.Tensor],
        graph: HeteroGraph,
    ) -> dict[str, torch.Tensor]:
        node_ids = graph.node_ids("global")
        global_embeddings = embeddings.get("global")
        if global_embeddings is None:
            return {}
        return {
            node_id: global_embeddings[idx]
            for idx, node_id in enumerate(node_ids)
            if idx < global_embeddings.size(0)
        }

    def pair_logits(
        self,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, torch.Tensor],
        graph: HeteroGraph,
    ) -> torch.Tensor:
        by_node = self._embedding_by_node(embeddings, graph)
        logits: list[torch.Tensor] = []
        device = next(self.parameters()).device
        for node_a, node_b in pairs:
            emb_a = by_node.get(node_a)
            emb_b = by_node.get(node_b)
            if emb_a is None or emb_b is None:
                logits.append(torch.zeros((), device=device))
                continue
            emb_a_proj = self.sim_proj(emb_a.unsqueeze(0)).squeeze(0)
            emb_b_proj = self.sim_proj(emb_b.unsqueeze(0)).squeeze(0)
            logits.append(5.0 * F.cosine_similarity(
                emb_a_proj.unsqueeze(0),
                emb_b_proj.unsqueeze(0),
                dim=-1,
            ).squeeze(0))
        if not logits:
            return torch.empty((0,), device=device)
        return torch.stack(logits)

    def pair_scores(
        self,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, torch.Tensor],
        graph: HeteroGraph,
    ) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(pairs, embeddings, graph))

    def resolve(
        self,
        node_a: str,
        node_b: str,
        embeddings: dict[str, torch.Tensor],
        graph: Optional[HeteroGraph] = None,
    ) -> SimilarityOutput:
        if graph is None:
            return SimilarityOutput(
                node_a=node_a,
                node_b=node_b,
                similarity=0.0,
                is_cold_start=False,
            )

        score = self.pair_scores([(node_a, node_b)], embeddings, graph)
        similarity = float(score.item()) if score.numel() else 0.0

        return SimilarityOutput(
            node_a=node_a,
            node_b=node_b,
            similarity=similarity,
            is_cold_start=False,
        )

    def resolve_batch(
        self,
        pairs: list[tuple[str, str]],
        embeddings: dict[str, torch.Tensor],
        graph: Optional[HeteroGraph] = None,
    ) -> list[SimilarityOutput]:
        return [
            self.resolve(a, b, embeddings, graph)
            for a, b in pairs
        ]
