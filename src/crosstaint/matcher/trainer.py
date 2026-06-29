"""Training pipeline for bridge event matcher with hard negative mining."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from crosstaint.matcher.model import BridgeEventMatcher, MatcherOutput
from crosstaint.types import DecodedEvent, EventFeatures, TaintPair


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for matcher training."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 50
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    triplet_margin: float = 0.5
    bce_weight: float = 0.7
    triplet_weight: float = 0.3
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 10


class TaintPairDataset(Dataset):
    """Dataset for taint tracking pairs.

    Stores positive pairs (matched source-destination events) and
    negative pairs (non-matching events) for training.
    """

    def __init__(
        self,
        pairs: List[TaintPair],
        features: Dict[str, EventFeatures],
        pos_neg_ratio: float = 1.0,
    ) -> None:
        """Initialize the dataset.

        Args:
            pairs: List of taint pairs
            features: Dictionary mapping event IDs to features
            pos_neg_ratio: Ratio of positive to negative samples
        """
        self.features = features
        self.pairs = pairs

        self.positive_pairs = [p for p in pairs if p.is_positive]
        self.negative_pairs = [p for p in pairs if not p.is_positive]

        num_negatives = int(len(self.positive_pairs) * pos_neg_ratio)
        self.sampled_negatives = self.negative_pairs[:num_negatives]

        self.dataset = self.positive_pairs + self.sampled_negatives
        self.labels = torch.tensor(
            [int(p.is_positive) for p in self.dataset],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self.dataset[idx]

        e1_features = self.features[pair.source_event_id]
        e2_features = self.features[pair.dest_event_id]

        e1_tensor = self._features_to_tensor(e1_features)
        e2_tensor = self._features_to_tensor(e2_features)

        return e1_tensor, e2_tensor, self.labels[idx]

    def _features_to_tensor(self, features: EventFeatures) -> torch.Tensor:
        parts = [
            features.selector_embedding,
            features.topics_embedding,
            np.array([features.value_bracket], dtype=np.float32),
            np.array([features.timestamp_feature], dtype=np.float32),
            features.context_embedding,
        ]
        concatenated = np.concatenate(parts)
        return torch.from_numpy(concatenated).float().unsqueeze(0).repeat(5, 1)


class HardNegativeMiner:
    """Mines hard negative examples using matcher confidence scores.

    Hard negatives are non-matching pairs that receive high similarity
    scores from the matcher, representing ambiguous cases.
    """

    def __init__(
        self,
        matcher: BridgeEventMatcher,
        device: torch.device,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize the hard negative miner.

        Args:
            matcher: The matcher model to use for scoring
            device: Device to run inference on
            confidence_threshold: Minimum confidence to consider as hard negative
        """
        self.matcher = matcher
        self.device = device
        self.confidence_threshold = confidence_threshold

    def mine(
        self,
        pairs: List[TaintPair],
        features: Dict[str, EventFeatures],
        k: int = 5,
    ) -> List[TaintPair]:
        positive_pairs = [p for p in pairs if p.is_positive]
        candidate_negatives = [p for p in pairs if not p.is_positive]

        if not positive_pairs or not candidate_negatives:
            return []

        positive_by_dest = {}
        for p in positive_pairs:
            key = (p.dest_bridge, p.dest_chain)
            if key not in positive_by_dest:
                positive_by_dest[key] = []
            positive_by_dest[key].append(p)

        scored_negatives = []
        for neg in candidate_negatives:
            key = (neg.dest_bridge, neg.dest_chain)
            if key not in positive_by_dest:
                continue

            pos_matches = positive_by_dest[key]
            for pos in pos_matches:
                e1_features = features.get(pos.source_event_id)
                e2_features = features.get(neg.dest_event_id)

                if e1_features is None or e2_features is None:
                    continue

                e1_tensor = self._features_to_tensor(e1_features).unsqueeze(0).to(self.device)
                e2_tensor = self._features_to_tensor(e2_features).unsqueeze(0).to(self.device)

                self.matcher.eval()
                with torch.no_grad():
                    output: MatcherOutput = self.matcher(e1_tensor, e2_tensor)
                    score = output.score.item()

                if score >= self.confidence_threshold:
                    scored_negatives.append((neg, score))
                    break

        scored_negatives.sort(key=lambda x: x[1], reverse=True)
        mined = [pair for pair, _ in scored_negatives[:k * len(positive_pairs)]]

        return mined[:len(positive_pairs)]

    def _features_to_tensor(self, features: EventFeatures) -> torch.Tensor:
        parts = [
            features.selector_embedding,
            features.topics_embedding,
            np.array([features.value_bracket], dtype=np.float32),
            np.array([features.timestamp_feature], dtype=np.float32),
            features.context_embedding,
        ]
        concatenated = np.concatenate(parts)
        return torch.from_numpy(concatenated).float().unsqueeze(0).repeat(5, 1)


class MatcherTrainer:
    """Training pipeline for bridge event matcher.

    Implements training loop with:
    - AdamW optimizer with weight decay
    - Cosine annealing learning rate schedule
    - Gradient clipping
    - Triplet loss + BCE loss combination
    - Best model checkpointing
    """

    def __init__(
        self,
        matcher: BridgeEventMatcher,
        train_pairs: List[TaintPair],
        val_pairs: List[TaintPair],
        features: Dict[str, EventFeatures],
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            matcher: BridgeEventMatcher model
            train_pairs: Training taint pairs
            val_pairs: Validation taint pairs
            features: Dictionary mapping event IDs to features
            config: Training configuration
            device: Device to train on
        """
        self.matcher = matcher
        self.features = features
        self.config = config or TrainingConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.matcher.to(self.device)

        self.train_dataset = TaintPairDataset(train_pairs, features)
        self.val_dataset = TaintPairDataset(val_pairs, features)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.optimizer = optim.AdamW(
            self.matcher.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.01,
        )

        self.bce_loss = nn.BCELoss()
        self.triplet_loss = nn.TripletMarginLoss(
            margin=self.config.triplet_margin,
            p=2,
        )

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.checkpoint_path = Path(self.config.checkpoint_dir)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.miner = HardNegativeMiner(self.matcher, self.device)

    def train(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, val_accuracy = self._validate(epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, "best_model.pt")
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break

            if epoch > 0 and epoch % 10 == 0:
                hard_negatives = self.miner.mine(
                    self.train_dataset.pairs + self.train_dataset.sampled_negatives,
                    self.features,
                    k=5,
                )
                if hard_negatives:
                    self._add_hard_negatives(hard_negatives)

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.matcher.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
            disable=not logging.getLogger().isEnabledFor(logging.INFO),
        )

        for e1, e2, labels in progress_bar:
            e1 = e1.to(self.device)
            e2 = e2.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            output = self.matcher(e1, e2, return_embeddings=True)

            bce = self.bce_loss(output.score, labels)

            triplet = self._compute_triplet_loss(output)

            loss = (
                self.config.bce_weight * bce +
                self.config.triplet_weight * triplet
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.matcher.parameters(),
                self.config.gradient_clip_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _compute_triplet_loss(self, output: MatcherOutput) -> torch.Tensor:
        if output.embedding1 is None or output.embedding2 is None:
            return torch.tensor(0.0, device=self.device)

        anchor = output.embedding1
        positive = output.embedding2

        batch_size = anchor.size(0)
        neg_indices = torch.randperm(batch_size, device=self.device)
        negative = anchor[neg_indices]

        return self.triplet_loss(anchor, positive, negative)

    def _validate(self, epoch: int) -> Tuple[float, float]:
        del epoch
        self.matcher.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for e1, e2, labels in self.val_loader:
                e1 = e1.to(self.device)
                e2 = e2.to(self.device)
                labels = labels.to(self.device)

                output = self.matcher(e1, e2)

                loss = self.bce_loss(output.score, labels)
                total_loss += loss.item()

                predictions = (output.score >= 0.5).float()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def predict(
        self,
        e1_features: EventFeatures,
        e2_features: EventFeatures,
    ) -> float:
        self.matcher.eval()

        e1_tensor = self._features_to_tensor(e1_features).unsqueeze(0).to(self.device)
        e2_tensor = self._features_to_tensor(e2_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.matcher(e1_tensor, e2_tensor)

        return output.score.item()

    def _features_to_tensor(self, features: EventFeatures) -> torch.Tensor:
        parts = [
            features.selector_embedding,
            features.topics_embedding,
            np.array([features.value_bracket], dtype=np.float32),
            np.array([features.timestamp_feature], dtype=np.float32),
            features.context_embedding,
        ]
        concatenated = np.concatenate(parts)
        return torch.from_numpy(concatenated).float().unsqueeze(0).repeat(5, 1)

    def _add_hard_negatives(self, hard_negatives: List[TaintPair]) -> None:
        for pair in hard_negatives:
            if pair not in self.train_dataset.sampled_negatives:
                self.train_dataset.sampled_negatives.append(pair)
                self.train_dataset.dataset.append(pair)
                self.train_dataset.labels = torch.cat([
                    self.train_dataset.labels,
                    torch.tensor([0.0]),
                ])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def _save_checkpoint(self, epoch: int, filename: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.matcher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, self.checkpoint_path / filename)

    def load_checkpoint(self, filename: str) -> int:
        checkpoint = torch.load(
            self.checkpoint_path / filename,
            map_location=self.device,
        )
        self.matcher.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        return checkpoint["epoch"]
