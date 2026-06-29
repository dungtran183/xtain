"""Featurizer for converting bridge events into transformer input format."""

from __future__ import annotations

import datetime
import hashlib
import random
from typing import Dict, List

import numpy as np
import torch

from crosstaint.types import DecodedEvent, EventFeatures


class EventFeaturizer:
    """Converts decoded bridge events into 5-token model inputs."""

    BRIDGE_FAMILIES: Dict[str, int] = {
        "wormhole": 0,
        "layerzero": 1,
        "multichain": 2,
        "stargate": 3,
        "across": 4,
        "hop": 5,
        "cbridge": 6,
        "synapse": 7,
        "hyperlane": 8,
    }

    HOURS_PER_WEEK = 7 * 24

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

        self._proj_matrix = self._generate_projection_matrix((64, 64), seed)

        self._context_weights = self._generate_context_weights(32, seed + 1)

        self._selector_weights = self._generate_projection_matrix((4, 4), seed + 2)

    def _generate_projection_matrix(
        self,
        shape: tuple[int, int],
        seed: int,
    ) -> np.ndarray:
        rng = random.Random(seed)
        matrix = np.array(
            [rng.uniform(-1, 1) for _ in range(shape[0] * shape[1])],
            dtype=np.float32,
        ).reshape(shape)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        return matrix

    def _generate_context_weights(self, dim: int, seed: int) -> np.ndarray:
        rng = random.Random(seed)
        weights = np.array(
            [rng.uniform(-0.5, 0.5) for _ in range(len(self.BRIDGE_FAMILIES) * dim)],
            dtype=np.float32,
        ).reshape(len(self.BRIDGE_FAMILIES), dim)
        return weights

    def featurize(self, event: DecodedEvent) -> EventFeatures:
        selector_emb = self._compute_selector_embedding(event.selector)

        topics_emb = self._compute_topics_embedding(event.topics)

        value_bracket = self._compute_value_bracket(event.value)

        timestamp_feat = self._compute_timestamp_feature(event.timestamp)

        context_emb = self._compute_context_embedding(event.bridge_family)

        return EventFeatures(
            selector_embedding=selector_emb,
            topics_embedding=topics_emb,
            value_bracket=value_bracket,
            timestamp_feature=timestamp_feat,
            context_embedding=context_emb,
        )

    def _compute_selector_embedding(self, selector: str) -> np.ndarray:
        selector_bytes = selector.encode("utf-8")[:4]
        while len(selector_bytes) < 4:
            selector_bytes += b"\x00"

        bucket = int.from_bytes(selector_bytes[:4], byteorder="big") % 4

        embedding = np.zeros(4, dtype=np.float32)
        embedding[bucket] = 1.0

        rotated = np.dot(embedding, self._selector_weights)
        return rotated

    def _compute_topics_embedding(self, topics: List[str]) -> np.ndarray:
        topic_hash = np.zeros(64, dtype=np.float32)

        for i, topic in enumerate(topics[:8]):
            topic_bytes = topic.encode("utf-8") if isinstance(topic, str) else bytes(topic)
            digest = hashlib.sha256(topic_bytes).digest()
            hash_val = int.from_bytes(digest[:4], byteorder="big")
            hash_float = (hash_val / (2**32 - 1)) * 2 - 1

            bucket_idx = i % 64
            topic_hash[bucket_idx] = hash_float

        projected = np.dot(topic_hash, self._proj_matrix)
        return np.tanh(projected)

    def _compute_value_bracket(self, value: int) -> float:
        if value <= 0:
            return 0.0

        log_value = np.log10(float(value))
        clamped = np.clip(log_value, 0.0, 19.0)
        bracket = clamped / 19.0

        return bracket

    def _compute_timestamp_feature(self, timestamp: object) -> float:
        if isinstance(timestamp, datetime.datetime):
            timestamp_seconds = int(timestamp.timestamp())
        else:
            timestamp_seconds = int(timestamp)
        hour_of_week = (timestamp_seconds // 3600) % self.HOURS_PER_WEEK
        return hour_of_week / self.HOURS_PER_WEEK

    def _compute_context_embedding(self, bridge_family: str) -> np.ndarray:
        family_lower = bridge_family.lower()
        family_idx = self.BRIDGE_FAMILIES.get(family_lower, 0)

        one_hot = np.zeros(len(self.BRIDGE_FAMILIES), dtype=np.float32)
        one_hot[family_idx] = 1.0

        context = np.dot(one_hot, self._context_weights)
        return context

    def featurize_batch(self, events: List[DecodedEvent]) -> List[EventFeatures]:
        return [self.featurize(event) for event in events]

    def event_to_tensor(self, features: EventFeatures) -> torch.Tensor:
        parts = [
            features.selector_embedding,
            features.topics_embedding,
            np.array([features.value_bracket], dtype=np.float32),
            np.array([features.timestamp_feature], dtype=np.float32),
            features.context_embedding,
        ]

        concatenated = np.concatenate(parts)
        tensor = torch.from_numpy(concatenated).float()

        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 5, 1)
        return tensor
