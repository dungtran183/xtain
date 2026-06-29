"""Cold-start fallback for zero-history addresses."""

from __future__ import annotations

from typing import Protocol

from crosstaint.types import SimilarityOutput


class MatcherProtocol(Protocol):
    """Minimal matcher interface required by cold-start fallback."""

    def match(self, source_event: str, dest_event: str) -> float:
        """Return confidence score in [0, 1] for a source-dest pair."""
        ...


class ColdStartFallback:
    """Fallback resolver for addresses with negligible history.

    Addresses with degree <= 2 at first observation lack sufficient graph
    structure for the HGT resolver. This class uses the matcher confidence
    as a proxy signal, applying a conservative threshold to avoid false links.

    Threshold rule:
        - If matcher_confidence >= 0.75: return similarity = matcher_confidence
        - Otherwise: return similarity = 0.0
    """

    CONFIDENCE_THRESHOLD: float = 0.75

    def resolve(
        self,
        node_a: str,
        node_b: str,
        matcher_confidence: float,
    ) -> SimilarityOutput:
        """Resolve a pair using matcher confidence as a cold-start proxy.

        Args:
            node_a: first node identifier
            node_b: second node identifier
            matcher_confidence: matcher confidence score in [0, 1]

        Returns:
            SimilarityOutput with is_cold_start=True and appropriate similarity
        """
        if matcher_confidence >= self.CONFIDENCE_THRESHOLD:
            similarity = float(matcher_confidence)
        else:
            similarity = 0.0

        return SimilarityOutput(
            node_a=node_a,
            node_b=node_b,
            similarity=similarity,
            is_cold_start=True,
            cold_start_matcher_confidence=float(matcher_confidence),
        )

    def resolve_batch(
        self,
        pairs: list[tuple[str, str]],
        matcher_confidences: list[float],
    ) -> list[SimilarityOutput]:
        """Batch cold-start resolution.

        Args:
            pairs: list of (node_a, node_b) tuples
            matcher_confidences: parallel list of matcher confidence scores

        Returns:
            list of SimilarityOutput, one per pair
        """
        if len(pairs) != len(matcher_confidences):
            raise ValueError("pairs and matcher_confidences must have the same length")

        return [
            self.resolve(a, b, conf)
            for (a, b), conf in zip(pairs, matcher_confidences)
        ]
