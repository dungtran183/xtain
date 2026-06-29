from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from crosstaint.config import Config
from crosstaint.propagation.decay import DecayScheduler
from crosstaint.propagation.weight import EdgeWeightLearner
from crosstaint.types import (
    AddressTag,
    EdgeType,
    IREdge,
    IRNode,
    NodeType,
    PropagationResult,
    PropagationState,
    SoundnessCertificate,
    SuspectEntry,
)


@dataclass
class _FrontierEntry:
    score: float
    node_id: str
    remaining_value: int
    hop_distance: int

    def __lt__(self, other: "_FrontierEntry") -> bool:
        return self.score > other.score


@dataclass
class _VisitedEdge:
    edge_id: str
    source: str
    target: str

    def __hash__(self) -> int:
        return hash((self.edge_id, self.source, self.target))


class PropagationEngine:
    def __init__(
        self,
        rho: float = 0.81,
        threshold: float = 0.04,
        time_window_days: int = 30,
        seed: int = 42,
        skip_hop_recovery: bool = True,
        skip_gate_multiplier: float = 2.0,
        max_hops: int = 11,
        mixer_addresses: frozenset[str] | None = None,
        edge_learner: EdgeWeightLearner | None = None,
        cold_start_degree: int = 2,
        beta_hat: float | None = None,
        finite_sample_epsilon: float | None = None,
        batch_window_blocks: int = 20,
    ) -> None:
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.rho = rho
        self.threshold = threshold
        self.time_window_days = time_window_days
        self.seed = seed
        self.skip_hop_recovery = skip_hop_recovery
        self.skip_gate = min(1.0, threshold * skip_gate_multiplier)
        self.max_hops = max_hops
        self.mixer_addresses = mixer_addresses or frozenset()
        self.cold_start_degree = cold_start_degree

        self.decay_scheduler = DecayScheduler(rho)
        self.edge_learner = edge_learner or EdgeWeightLearner()
        self.beta_hat = beta_hat
        self.finite_sample_epsilon = finite_sample_epsilon
        self.batch_window_blocks = max(1, batch_window_blocks)

    @classmethod
    def from_config(cls, config: Config | None = None, **overrides: object) -> "PropagationEngine":
        cfg = config or Config.load()
        propagation = cfg.propagation
        mixer_cfg = propagation.get("mixer_detection", {})
        known_mixers = mixer_cfg.get("known_mixers", []) if isinstance(mixer_cfg, dict) else []
        mixer_addresses = frozenset(
            str(address).lower()
            for address in known_mixers
            if isinstance(address, str)
        )
        cold_start_degree = int(cfg.pseudonym.get("cold_start_threshold", 2))

        return cls(
            rho=float(overrides.get("rho", propagation.get("rho", 0.81))),
            threshold=float(overrides.get("threshold", propagation.get("threshold", 0.04))),
            time_window_days=int(propagation.get("time_window_days", 30)),
            seed=int(overrides.get("seed", propagation.get("seed", 42))),
            max_hops=int(propagation.get("max_hops", 11)),
            mixer_addresses=mixer_addresses,
            edge_learner=EdgeWeightLearner.from_config(propagation),
            cold_start_degree=cold_start_degree,
            beta_hat=_optional_float(
                overrides.get("beta_hat", propagation.get("soundness", {}).get("beta_hat"))
            ),
            finite_sample_epsilon=_optional_float(
                overrides.get(
                    "finite_sample_epsilon",
                    propagation.get("soundness", {}).get("finite_sample_epsilon"),
                )
            ),
            batch_window_blocks=int(
                overrides.get(
                    "batch_window_blocks",
                    propagation.get("soundness", {}).get("batch_window_blocks", 20),
                )
            ),
        )

    def _is_mixer(self, node: IRNode) -> bool:
        address_lower = node.address.lower()
        return (
            node.node_type == NodeType.MIXER
            or address_lower in self.mixer_addresses
            or AddressTag.MIXER_DEPOSIT in node.tags
        )

    def _init_frontier(
        self,
        origin: str,
        origin_chain: str,
        origin_value: int,
    ) -> PropagationState:
        entry = _FrontierEntry(
            score=1.0,
            node_id=origin,
            remaining_value=origin_value,
            hop_distance=0,
        )
        return PropagationState(
            frontier=[entry],
            scores={origin: 1.0},
            predecessors={},
            hop_distances={origin: 0},
            terminated=set(),
            visited_edges=set(),
            values={origin: origin_value},
        )

    def _value_conserved(self, edge: IREdge, remaining_value: int) -> bool:
        if remaining_value <= 0 or edge.value <= 0:
            return False
        fee_bound_pct = max(0.0, edge.fee_bound_pct)
        fee_bound = math.ceil(remaining_value * fee_bound_pct)
        return edge.value <= remaining_value + fee_bound

    def _value_conserved_skip(self, edge: IREdge, remaining_value: int) -> bool:
        if remaining_value <= 0 or edge.value <= 0:
            return False
        fee_bound_pct = max(0.0, edge.fee_bound_pct)
        composed_bound = math.ceil(remaining_value * min(1.0, 2.0 * fee_bound_pct))
        return edge.value <= remaining_value + composed_bound

    def _soundness_certificate(self, edges: list[IREdge]) -> SoundnessCertificate:
        cross_chain_edges = [
            edge for edge in edges
            if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE
        ]
        groups = {
            (
                edge.bridge,
                edge.block_number // self.batch_window_blocks
                if edge.block_number is not None
                else 0,
            )
            for edge in cross_chain_edges
        }
        batch_groups = max(1, len(groups))
        parameters_loaded = (
            self.beta_hat is not None
            and self.finite_sample_epsilon is not None
        )
        bound = None
        if parameters_loaded:
            bound = min(
                1.0,
                batch_groups * (float(self.beta_hat) + float(self.finite_sample_epsilon)),
            )
        return SoundnessCertificate(
            bound=bound,
            beta_hat=self.beta_hat,
            finite_sample_epsilon=self.finite_sample_epsilon,
            batch_groups=batch_groups,
            parameters_loaded=parameters_loaded,
        )

    def _matcher_confidence(self, edge: IREdge, matcher: Any | None) -> float:
        if edge.edge_type != EdgeType.CROSS_CHAIN_BRIDGE:
            return 1.0
        if matcher is None:
            return 1.0
        if hasattr(matcher, "score_edge"):
            score = matcher.score_edge(edge)
        elif hasattr(matcher, "match") and edge.source_event and edge.target_event:
            score = matcher.match(edge.source_event, edge.target_event)
        elif callable(matcher):
            score = matcher(edge)
        else:
            score = 1.0
        return float(min(1.0, max(0.0, score)))

    def _resolver_multiplier(
        self,
        source: str,
        target: str,
        graph: dict[str, IRNode],
        resolver: Any | None,
        matcher_confidence: float,
    ) -> float:
        target_node = graph.get(target)
        if target_node is None:
            return 1.0
        if target_node.degree_at_first_seen <= self.cold_start_degree:
            return matcher_confidence
        if resolver is None or not hasattr(resolver, "resolve"):
            return 1.0
        output = resolver.resolve(source, target)
        similarity = getattr(output, "similarity", 1.0)
        return float(min(1.0, max(0.0, similarity)))

    def _step(
        self,
        state: PropagationState,
        graph: dict[str, IRNode],
        outgoing_by_source: dict[str, list[IREdge]],
        matcher: Any | None,
        resolver: Any | None,
    ) -> bool:
        if not state.frontier:
            return False

        entry: _FrontierEntry = heapq.heappop(state.frontier)
        score = entry.score
        node_id = entry.node_id
        remaining_value = entry.remaining_value
        hop_distance = entry.hop_distance

        if score < state.scores.get(node_id, 0.0):
            return bool(state.frontier)

        if score < self.threshold:
            return bool(state.frontier)

        if node_id in state.terminated:
            return bool(state.frontier)

        node = graph.get(node_id)
        if node is not None and self._is_mixer(node):
            state.terminated.add(node_id)
            return bool(state.frontier)

        has_cross_chain_exit = False
        matched_cross_chain = False
        for edge in outgoing_by_source.get(node_id, []):
            if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE:
                has_cross_chain_exit = True
            edge_key = _VisitedEdge(edge.edge_id, edge.source_node, edge.target_node)
            if edge_key in state.visited_edges:
                continue
            state.visited_edges.add(edge_key)

            if not self._value_conserved(edge, remaining_value):
                continue

            edge_weight = self.edge_learner.get_weight(edge)
            matcher_confidence = self._matcher_confidence(edge, matcher)
            resolver_weight = self._resolver_multiplier(
                edge.source_node,
                edge.target_node,
                graph,
                resolver,
                matcher_confidence,
            )
            hop = hop_distance + 1
            if hop > self.max_hops:
                continue
            decayed_score = score * self.rho * edge_weight * matcher_confidence * resolver_weight
            new_value = int(edge.value)

            if decayed_score >= self.threshold:
                target = edge.target_node
                if edge.edge_type == EdgeType.CROSS_CHAIN_BRIDGE:
                    matched_cross_chain = True
                if (
                    target not in state.scores
                    or decayed_score > state.scores[target]
                ):
                    state.scores[target] = decayed_score
                    state.hop_distances[target] = hop
                    state.values[target] = new_value
                    state.predecessors[target] = node_id
                    heapq.heappush(
                        state.frontier,
                        _FrontierEntry(
                            score=decayed_score,
                            node_id=target,
                            remaining_value=new_value,
                            hop_distance=hop,
                        ),
                    )

        if (
            self.skip_hop_recovery
            and has_cross_chain_exit
            and not matched_cross_chain
        ):
            self._skip_hop(
                state,
                graph,
                outgoing_by_source,
                matcher,
                resolver,
                node_id,
                score,
                remaining_value,
                hop_distance,
            )

        return bool(state.frontier)

    def _skip_hop(
        self,
        state: PropagationState,
        graph: dict[str, IRNode],
        outgoing_by_source: dict[str, list[IREdge]],
        matcher: Any | None,
        resolver: Any | None,
        node_id: str,
        score: float,
        remaining_value: int,
        hop_distance: int,
    ) -> None:
        # Bounded single-hop reconnection: when no direct cross-chain edge cleared
        # the gate, search two-hop continuations a -> x -> b' that satisfy the
        # composed value envelope and clear the stricter skip gate. The recovery
        # never chains skips, so it cannot manufacture long speculative trails.
        for first_edge in outgoing_by_source.get(node_id, []):
            intermediate = first_edge.target_node
            for second_edge in outgoing_by_source.get(intermediate, []):
                if second_edge.edge_type != EdgeType.CROSS_CHAIN_BRIDGE:
                    continue
                target = second_edge.target_node
                if target == node_id:
                    continue
                if not self._value_conserved_skip(second_edge, remaining_value):
                    continue
                edge_key = _VisitedEdge(
                    f"skip:{first_edge.edge_id}:{second_edge.edge_id}",
                    node_id,
                    target,
                )
                if edge_key in state.visited_edges:
                    continue
                state.visited_edges.add(edge_key)

                edge_weight = self.edge_learner.get_weight(second_edge)
                matcher_confidence = self._matcher_confidence(second_edge, matcher)
                resolver_weight = self._resolver_multiplier(
                    node_id,
                    target,
                    graph,
                    resolver,
                    matcher_confidence,
                )
                hop = hop_distance + 2
                if hop > self.max_hops:
                    continue
                decayed_score = (
                    score
                    * (self.rho ** 2)
                    * edge_weight
                    * matcher_confidence
                    * resolver_weight
                )
                new_value = int(second_edge.value)

                if decayed_score > self.skip_gate and (
                    target not in state.scores
                    or decayed_score > state.scores[target]
                ):
                    state.scores[target] = decayed_score
                    state.hop_distances[target] = hop
                    state.values[target] = new_value
                    state.predecessors[target] = node_id
                    heapq.heappush(
                        state.frontier,
                        _FrontierEntry(
                            score=decayed_score,
                            node_id=target,
                            remaining_value=new_value,
                            hop_distance=hop,
                        ),
                    )

    def propagate(
        self,
        origin: str,
        origin_chain: str,
        origin_value: int,
        graph: dict[str, IRNode],
        edges: list[IREdge],
        case_id: str,
        matcher: Any | None = None,
        resolver: Any | None = None,
    ) -> PropagationResult:
        start_time = time.monotonic()

        state = self._init_frontier(origin, origin_chain, origin_value)
        path_count = 0
        outgoing_by_source: dict[str, list[IREdge]] = defaultdict(list)
        for edge in edges:
            outgoing_by_source[edge.source_node].append(edge)

        while self._step(state, graph, outgoing_by_source, matcher, resolver):
            path_count += 1

        runtime_ms = (time.monotonic() - start_time) * 1000

        suspect_entries: list[SuspectEntry] = []
        for node_id, score in state.scores.items():
            node = graph.get(node_id)
            if node is None:
                continue
            if node_id == origin:
                continue

            pred = state.predecessors.get(node_id, "")
            hop = state.hop_distances.get(node_id, 0)
            flags: set[str] = set()
            if node_id in state.terminated:
                flags.add("mixer_terminated")

            suspect_entries.append(
                SuspectEntry(
                    address=node.address,
                    chain=node.chain,
                    taint_score=score,
                    hop_distance=hop,
                    predecessors=(pred,) if pred else (),
                    is_cold_start=node.degree_at_first_seen <= self.cold_start_degree,
                    flags=frozenset(flags),
                )
            )

        suspect_entries.sort(key=lambda e: e.taint_score, reverse=True)

        config_hash = Config.config_hash() if Config._instance is not None else ""

        return PropagationResult(
            case_id=case_id,
            origin=origin,
            origin_chain=origin_chain,
            origin_value=origin_value,
            suspect_set=tuple(suspect_entries),
            runtime_ms=runtime_ms,
            config_hash=config_hash,
            seed=self.seed,
            path_count=path_count,
            soundness_certificate=self._soundness_certificate(edges),
        )


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
