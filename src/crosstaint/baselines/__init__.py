"""Deterministic baselines for CrossTaint evaluation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from crosstaint.types import EdgeType, PropagationResult, SuspectEntry


@dataclass(frozen=True, slots=True)
class _GraphCase:
    origin: str
    origin_chain: str
    origin_value: int
    graph: dict[str, Any]


class _GraphView:
    def __init__(self, graph: dict[str, Any]) -> None:
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])
        self.node_by_id = {
            node.get("node_id", f"{node.get('chain')}:{node.get('address', '').lower()}"): node
            for node in self.nodes
        }

    def outgoing(self, node_id: str) -> list[dict[str, Any]]:
        return [edge for edge in self.edges if edge.get("source") == node_id]

    def node_address(self, node_id: str) -> str:
        node = self.node_by_id.get(node_id)
        if node is None:
            return node_id.split(":", 1)[1] if ":" in node_id else node_id
        return str(node.get("address", node_id))

    def node_chain(self, node_id: str) -> str:
        node = self.node_by_id.get(node_id)
        if node is None:
            return node_id.split(":", 1)[0] if ":" in node_id else "unknown"
        return str(node.get("chain", "unknown"))


class _TraversalBaseline:
    name = "TraversalBaseline"

    def __init__(self, max_hops: int = 5) -> None:
        self.max_hops = max_hops
        self._fitted = False

    def fit(self, train_data: Any) -> None:
        self._fitted = True

    def _edge_allowed(
        self,
        edge: dict[str, Any],
        previous_edge: dict[str, Any] | None,
        current_value: int,
    ) -> bool:
        return True

    def _score(self, hop_distance: int, edge: dict[str, Any]) -> float:
        edge_type = edge.get("edge_type")
        weight = 0.9 if edge_type == EdgeType.CROSS_CHAIN_BRIDGE else 1.0
        return weight / (hop_distance + 1)

    def predict(self, case_data: Any) -> PropagationResult:
        start_time = time.perf_counter()

        origin = getattr(case_data, "origin")
        origin_chain = getattr(case_data, "origin_chain", "unknown")
        origin_value = int(getattr(case_data, "origin_value", 0))
        graph = _GraphView(getattr(case_data, "graph", {}))

        queue = deque([(origin, 0, origin_value, None)])
        visited: set[str] = set()
        suspect_set: list[SuspectEntry] = []

        while queue:
            node_id, hop_distance, value, previous_edge = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            if node_id != origin:
                suspect_set.append(
                    SuspectEntry(
                        address=graph.node_address(node_id),
                        chain=graph.node_chain(node_id),
                        taint_score=self._score(hop_distance, previous_edge or {}),
                        hop_distance=hop_distance,
                        predecessors=(origin,),
                        is_cold_start=False,
                        flags=frozenset([self.name]),
                    )
                )

            if hop_distance >= self.max_hops:
                continue

            for edge in graph.outgoing(node_id):
                if not self._edge_allowed(edge, previous_edge, value):
                    continue
                target = str(edge.get("target"))
                if target not in visited:
                    next_value = int(edge.get("value", value))
                    queue.append((target, hop_distance + 1, next_value, edge))

        runtime_ms = (time.perf_counter() - start_time) * 1000
        case_id = uuid5(NAMESPACE_URL, f"{self.name}:{origin}:{origin_chain}")

        suspect_set.sort(key=lambda entry: entry.taint_score, reverse=True)
        return PropagationResult(
            case_id=str(case_id),
            origin=origin,
            origin_chain=origin_chain,
            origin_value=origin_value,
            suspect_set=tuple(suspect_set),
            runtime_ms=runtime_ms,
            config_hash=self.name,
            seed=0,
            path_count=len(suspect_set),
        )


class NaiveBaseline(_TraversalBaseline):
    name = "NaiveBaseline"

    def _edge_allowed(
        self,
        edge: dict[str, Any],
        previous_edge: dict[str, Any] | None,
        current_value: int,
    ) -> bool:
        del previous_edge, current_value
        return edge.get("edge_type") == EdgeType.INTRA_CHAIN_TRANSFER


class HeuristicBaseline(_TraversalBaseline):
    name = "HeuristicBaseline"

    def __init__(
        self,
        max_hops: int = 8,
        value_ratio_threshold: float = 10.0,
    ) -> None:
        super().__init__(max_hops=max_hops)
        self.value_ratio_threshold = value_ratio_threshold

    def _edge_allowed(
        self,
        edge: dict[str, Any],
        previous_edge: dict[str, Any] | None,
        current_value: int,
    ) -> bool:
        edge_value = int(edge.get("value", current_value))
        if current_value > 0 and edge_value > 0:
            ratio = max(current_value, edge_value) / min(current_value, edge_value)
            if ratio > self.value_ratio_threshold:
                return False
        if previous_edge is None:
            return True
        previous_bridge = previous_edge.get("bridge")
        current_bridge = edge.get("bridge")
        return previous_bridge == current_bridge or current_bridge is not None


class IntraChainCutoffBaseline(_TraversalBaseline):
    name = "IntraChainCutoffBaseline"

    def __init__(self, max_hops: int = 5, value_ratio_threshold: float = 100.0) -> None:
        super().__init__(max_hops=max_hops)
        self.value_ratio_threshold = value_ratio_threshold

    def _edge_allowed(
        self,
        edge: dict[str, Any],
        previous_edge: dict[str, Any] | None,
        current_value: int,
    ) -> bool:
        del previous_edge
        if edge.get("edge_type") == EdgeType.CROSS_CHAIN_BRIDGE:
            return False
        edge_value = int(edge.get("value", current_value))
        if current_value <= 0 or edge_value <= 0:
            return True
        return max(current_value, edge_value) / min(current_value, edge_value) <= self.value_ratio_threshold


class GraphDegreeBaseline:
    name = "GraphDegreeBaseline"

    def __init__(self, score_threshold: float = 0.35) -> None:
        self.score_threshold = score_threshold
        self._fitted = False

    def fit(self, train_data: Any) -> None:
        self._fitted = True

    def predict(self, case_data: Any) -> PropagationResult:
        start_time = time.perf_counter()

        origin = getattr(case_data, "origin")
        origin_chain = getattr(case_data, "origin_chain", "unknown")
        origin_value = int(getattr(case_data, "origin_value", 0))
        graph = _GraphView(getattr(case_data, "graph", {}))

        suspect_set: list[SuspectEntry] = []
        for node_id, node in graph.node_by_id.items():
            if node_id == origin:
                continue
            degree = int(node.get("degree", 0))
            bridge_count = int(node.get("bridge_count", 0))
            node_type = str(node.get("node_type", "EOA"))
            score = min(1.0, 0.08 * degree + 0.12 * bridge_count + (0.15 if node_type == "CONTRACT" else 0.0))
            if score >= self.score_threshold:
                suspect_set.append(
                    SuspectEntry(
                        address=graph.node_address(node_id),
                        chain=graph.node_chain(node_id),
                        taint_score=score,
                        hop_distance=1,
                        predecessors=(origin,),
                        is_cold_start=False,
                        flags=frozenset([self.name]),
                    )
                )

        runtime_ms = (time.perf_counter() - start_time) * 1000
        case_id = uuid5(NAMESPACE_URL, f"{self.name}:{origin}:{origin_chain}")
        suspect_set.sort(key=lambda entry: entry.taint_score, reverse=True)
        return PropagationResult(
            case_id=str(case_id),
            origin=origin,
            origin_chain=origin_chain,
            origin_value=origin_value,
            suspect_set=tuple(suspect_set),
            runtime_ms=runtime_ms,
            config_hash=self.name,
            seed=0,
            path_count=len(suspect_set),
        )
