"""
Type definitions for the CrossTaint intermediate representation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol


class NodeType:
    """Node type constants."""
    EOA = "EOA"
    CONTRACT = "CONTRACT"
    BRIDGE = "BRIDGE"
    MIXER = "MIXER"


class EdgeType:
    """Edge type constants."""
    INTRA_CHAIN_TRANSFER = "INTRA_CHAIN_TRANSFER"
    CROSS_CHAIN_BRIDGE = "CROSS_CHAIN_BRIDGE"
    DEX_SWAP = "DEX_SWAP"


class TaintOperator:
    """Taint operator constants."""
    LOCK = "Lock"
    MINT = "Mint"
    BURN = "Burn"
    RELEASE = "Release"
    INTENT_FILL = "IntentFill"


class BridgeMode:
    """Bridge mode constants."""
    LOCK_MINT = "lock_mint"
    BURN_MINT = "burn_mint"
    POOL = "pool"
    INTENT = "intent"


class AddressTag:
    """Address tag constants."""
    KNOWN_ATTACKER = "KNOWN_ATTACKER"
    MIXER_DEPOSIT = "MIXER_DEPOSIT"


@dataclass(frozen=True, slots=True)
class DecodedEvent:
    """Canonical representation of a decoded blockchain event."""
    event_id: str
    chain: str
    block_number: int
    timestamp: Any
    tx_hash: str
    event_name: str
    emitter_address: str
    topics: tuple[bytes, ...]
    params: dict[str, Any]
    bridge: str | None = None
    selector: str = ""
    value: int = 0
    bridge_family: str = ""


@dataclass(frozen=True, slots=True)
class IRNode:
    """Node in the unified intermediate representation graph."""
    node_id: str
    address: str
    chain: str
    node_type: str
    tags: frozenset[str]
    first_seen: Any | None
    degree_at_first_seen: int
    in_value_total: int
    out_value_total: int
    bridge_count: int
    dex_swap_count: int


@dataclass(frozen=True, slots=True)
class IREdge:
    """Edge in the unified intermediate representation graph."""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: str
    operator: str
    bridge: str
    source_event: str | None
    target_event: str | None
    value: int
    asset: str | None
    timestamp: Any
    block_number: int
    fee_bound_pct: float


@dataclass(frozen=True, slots=True)
class SyntheticHop:
    """A single hop in a synthetic cross-chain trajectory."""
    from_chain: str
    to_chain: str
    bridge: str
    operator: str
    value: int
    asset: str
    inter_arrival_seconds: float


@dataclass(frozen=True, slots=True)
class SyntheticTrajectory:
    """A complete synthetic cross-chain trajectory."""
    trajectory_id: uuid.UUID
    hops: tuple[SyntheticHop, ...]
    origin_address: str
    addresses: tuple[str, ...]
    total_hops: int
    total_value: int
    salt: str


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Evaluation metrics for a single run."""
    hop_recall: float
    precision: float
    false_positive_rate: float
    true_positive: int
    false_positive: int
    false_negative: int
    runtime_median_ms: float
    runtime_p95_ms: float
    num_cases: int


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""
    metric_name: str
    mean: float
    ci_lower: float
    ci_upper: float
    n_resamples: int
    alpha: float


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """Result of a Wilcoxon signed-rank test."""
    method_a: str
    method_b: str
    statistic: float
    p_value: float
    n_pairs: int
    significant: bool


@dataclass(frozen=True, slots=True)
class BoundEstimate:
    """Empirical estimate of bounded-mismatch parameters."""
    delta_hat: float
    delta_ci_lower: float
    delta_ci_upper: float
    beta_hat: float
    beta_ci_lower: float
    beta_ci_upper: float
    n_edges: int
    n_resamples: int
    predicted_offtrace_rate_independent: float
    observed_offtrace_rate: float
    n_batch_groups: int | None


@dataclass(frozen=True, slots=True)
class SoundnessCertificate:
    """Per-case bounded-mismatch certificate metadata."""
    bound: float | None
    beta_hat: float | None
    finite_sample_epsilon: float | None
    batch_groups: int
    parameters_loaded: bool


@dataclass(frozen=True, slots=True)
class PropagationResult:
    """Result of a taint propagation run."""
    case_id: str
    origin: str
    origin_chain: str
    origin_value: int
    suspect_set: tuple[Any, ...]
    runtime_ms: float
    config_hash: str
    seed: int
    path_count: int
    soundness_certificate: SoundnessCertificate | None = None


class BaselineProtocol(Protocol):
    """Protocol for baseline methods."""
    @property
    def name(self) -> str:
        ...

    def fit(self, train_data: Any) -> None:
        ...

    def predict(self, case_data: Any) -> PropagationResult:
        ...


@dataclass(frozen=True, slots=True)
class SuspectEntry:
    """Output entry in the final suspect set."""
    address: str
    chain: str
    taint_score: float
    hop_distance: int
    predecessors: tuple[str, ...]
    is_cold_start: bool
    flags: frozenset[str]


@dataclass(frozen=True, slots=True)
class RawEvent:
    """Raw event from blockchain RPC."""
    event_id: str
    chain: str
    block_number: int
    timestamp: Any
    tx_hash: str
    event_name: str
    emitter_address: str
    topics: tuple[bytes, ...]
    params: dict[str, Any]
    bridge: str | None = None


class ChainClientProtocol(Protocol):
    """Protocol for chain clients."""
    async def get_block_range(self, start: int, end: int) -> list[RawEvent]:
        ...


class BridgeAdapterProtocol(Protocol):
    """Protocol for bridge adapters."""
    def bridge_id(self) -> str:
        ...

    def bridge_mode(self) -> str:
        ...

    def source_event_names(self) -> tuple[str, ...]:
        ...

    def dest_event_names(self) -> tuple[str, ...]:
        ...

    def is_source_event(self, event: DecodedEvent) -> bool:
        ...

    def is_dest_event(self, event: DecodedEvent) -> bool:
        ...

    def decode_lock(self, event: DecodedEvent) -> IREdge | None:
        ...

    def decode_mint(self, event: DecodedEvent) -> IREdge | None:
        ...

    def decode_burn(self, event: DecodedEvent) -> IREdge | None:
        ...

    def decode_release(self, event: DecodedEvent) -> IREdge | None:
        ...

    def decode_intent_fill(self, event: DecodedEvent) -> IREdge | None:
        ...

    def is_pair(self, source: DecodedEvent, dest: DecodedEvent) -> bool:
        ...


EventId = str


@dataclass(frozen=True, slots=True)
class PropagationState:
    """Mutable state maintained during a propagation run."""
    frontier: list
    scores: dict
    predecessors: dict
    hop_distances: dict
    terminated: set
    visited_edges: set
    values: dict


@dataclass(frozen=True, slots=True)
class SimilarityOutput:
    """Output from pseudonym resolver similarity computation."""
    node_a: str
    node_b: str
    similarity: float
    is_cold_start: bool
    cold_start_matcher_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class EventFeatures:
    """Feature vector for a single bridge event used by the matcher."""
    selector_embedding: Any
    topics_embedding: Any
    value_bracket: float
    timestamp_feature: float
    context_embedding: Any


@dataclass(frozen=True, slots=True)
class NodeFeatures:
    """Feature vector for a graph node used by the pseudonym resolver."""
    in_value: int
    out_value: int
    bridge_count: int
    dex_swap_count: int
    degree: int
    degree_at_first_seen: int
    chain: str
    node_type: str

    def to_vector(self) -> list[float]:
        """Convert features to a flat list for GNN input."""
        return [
            float(self.in_value) / 1e22 if self.in_value else 0.0,
            float(self.out_value) / 1e22 if self.out_value else 0.0,
            float(self.bridge_count) / 100.0,
            float(self.dex_swap_count) / 100.0,
            float(self.degree) / 100.0,
            float(self.degree_at_first_seen) / 10.0,
            1.0 if self.chain == "ethereum" else 0.0,
            1.0 if self.node_type == "CONTRACT" else 0.0,
        ]


@dataclass(frozen=True, slots=True)
class TaintPair:
    """Labeled source-destination event pair for matcher training."""
    pair_id: str
    source_event_id: str
    dest_event_id: str
    is_positive: bool
    dest_bridge: str
    dest_chain: str
    source_value: int
    dest_value: int
    time_delta_seconds: float | None = None
