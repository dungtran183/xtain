"""Benchmark corpus loading and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import yaml

from crosstaint.config import Config
from crosstaint.types import AddressTag, EdgeType, IREdge, IRNode, NodeType, TaintOperator


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    origin_address: str
    origin_chain: str
    origin_value: int
    ground_truth_addresses: list[str]
    source_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def origin_node_id(self) -> str:
        return node_id(self.origin_chain, self.origin_address)


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    cases: list[BenchmarkCase]
    nodes: dict[str, IRNode]
    edges: list[IREdge]
    benign_addresses: list[str]
    manifest_hash: str
    metadata: dict[str, Any]


def node_id(chain: str, address: str) -> str:
    return f"{chain}:{address.lower()}"


def load_benchmark_dataset(
    manifest_path: Path,
    benign_addresses_path: Path | None = None,
    require_two_source_ground_truth: bool = True,
) -> BenchmarkDataset:
    manifest = _load_structured_file(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("benchmark manifest must be a JSON/YAML object")

    cases_raw = manifest.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("benchmark manifest must contain a non-empty cases list")

    nodes: dict[str, IRNode] = {}
    edges: dict[str, IREdge] = {}

    _load_graph_section(manifest.get("graph", {}), nodes, edges)

    cases: list[BenchmarkCase] = []
    for idx, raw_case in enumerate(cases_raw):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {idx} must be an object")
        case = _parse_case(raw_case, idx, nodes, edges, require_two_source_ground_truth)
        cases.append(case)

    benign_addresses = _load_benign_addresses(manifest, benign_addresses_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
    metadata = {
        "manifest_file": manifest_path.name,
        "benchmark_name": manifest.get("benchmark_name", ""),
        "version": manifest.get("version", ""),
    }

    return BenchmarkDataset(
        cases=cases,
        nodes=nodes,
        edges=list(edges.values()),
        benign_addresses=benign_addresses,
        manifest_hash=manifest_hash,
        metadata=metadata,
    )


def _load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        if suffix == ".json":
            return json.load(handle)
    raise ValueError(f"unsupported manifest format: {path}")


def _load_graph_section(
    graph_raw: Any,
    nodes: dict[str, IRNode],
    edges: dict[str, IREdge],
) -> None:
    if graph_raw in (None, {}):
        return
    if not isinstance(graph_raw, dict):
        raise ValueError("graph section must be an object")

    for raw_node in graph_raw.get("nodes", []):
        node = _parse_node(raw_node)
        nodes[node.node_id] = node

    for raw_edge in graph_raw.get("edges", []):
        edge = _parse_edge(raw_edge)
        edges[edge.edge_id] = edge
        _ensure_endpoint_nodes(edge, nodes)


def _parse_case(
    raw_case: dict[str, Any],
    case_index: int,
    nodes: dict[str, IRNode],
    edges: dict[str, IREdge],
    require_two_source_ground_truth: bool,
) -> BenchmarkCase:
    origin = raw_case.get("origin", {})
    if not isinstance(origin, dict):
        raise ValueError(f"case {case_index} origin must be an object")

    origin_address = _required_str(origin, "address", f"case {case_index} origin")
    origin_chain = _required_str(origin, "chain", f"case {case_index} origin")
    origin_value = int(origin.get("value", raw_case.get("origin_value", 0)))
    if origin_value < 0:
        raise ValueError(f"case {case_index} origin value must be non-negative")

    case_id = str(raw_case.get("case_id") or uuid5(NAMESPACE_URL, f"{origin_chain}:{origin_address}:{case_index}"))
    hops_raw = raw_case.get("hops")
    if not isinstance(hops_raw, list) or not hops_raw:
        raise ValueError(f"case {case_id} must contain a non-empty hops list")

    case_sources = raw_case.get("ground_truth_sources", raw_case.get("sources", []))
    source_count = len(case_sources) if isinstance(case_sources, list) else 0
    current_chain = origin_chain
    current_address = origin_address
    ground_truth: list[str] = []

    _ensure_node(nodes, origin_chain, origin_address, frozenset([AddressTag.KNOWN_ATTACKER]))

    for hop_index, hop in enumerate(hops_raw):
        if not isinstance(hop, dict):
            raise ValueError(f"case {case_id} hop {hop_index} must be an object")
        hop_sources = hop.get("sources", [])
        hop_source_count = len(hop_sources) if isinstance(hop_sources, list) else 0
        if require_two_source_ground_truth and max(source_count, hop_source_count) < 2:
            raise ValueError(f"case {case_id} hop {hop_index} lacks two-source ground truth")

        from_chain = str(hop.get("from_chain", current_chain))
        to_chain = _required_str(hop, "to_chain", f"case {case_id} hop {hop_index}")
        bridge = _required_str(hop, "bridge", f"case {case_id} hop {hop_index}")
        address = _required_str(hop, "address", f"case {case_id} hop {hop_index}")
        value = int(hop.get("value", origin_value))
        asset = hop.get("asset")
        operator = str(hop.get("operator", TaintOperator.LOCK))

        source_id = node_id(from_chain, current_address)
        target_id = node_id(to_chain, address)
        _ensure_node(nodes, from_chain, current_address, frozenset([AddressTag.KNOWN_ATTACKER]))
        _ensure_node(nodes, to_chain, address, frozenset([AddressTag.KNOWN_ATTACKER]))

        edge = IREdge(
            edge_id=_case_edge_id(case_id, hop_index, source_id, target_id, bridge),
            source_node=source_id,
            target_node=target_id,
            edge_type=EdgeType.CROSS_CHAIN_BRIDGE if from_chain != to_chain else EdgeType.INTRA_CHAIN_TRANSFER,
            operator=operator,
            bridge=bridge,
            source_event=hop.get("source_event"),
            target_event=hop.get("target_event"),
            value=value,
            asset=str(asset) if asset is not None else None,
            timestamp=hop.get("timestamp", raw_case.get("timestamp", "")),
            block_number=int(hop.get("block_number", 0)),
            fee_bound_pct=_fee_bound_pct(bridge, hop),
        )
        edges[edge.edge_id] = edge
        ground_truth.append(address)
        current_chain = to_chain
        current_address = address

    return BenchmarkCase(
        case_id=case_id,
        origin_address=origin_address,
        origin_chain=origin_chain,
        origin_value=origin_value,
        ground_truth_addresses=ground_truth,
        source_count=source_count,
        metadata={
            "name": raw_case.get("name", ""),
            "incident_date": raw_case.get("incident_date", ""),
            "sources": case_sources if isinstance(case_sources, list) else [],
        },
    )


def _parse_node(raw_node: Any) -> IRNode:
    if not isinstance(raw_node, dict):
        raise ValueError("graph node must be an object")
    chain = _required_str(raw_node, "chain", "graph node")
    address = _required_str(raw_node, "address", "graph node")
    tags_raw = raw_node.get("tags", [])
    tags = frozenset(str(tag) for tag in tags_raw) if isinstance(tags_raw, list) else frozenset()
    return IRNode(
        node_id=str(raw_node.get("node_id") or node_id(chain, address)),
        address=address.lower(),
        chain=chain,
        node_type=str(raw_node.get("node_type", NodeType.EOA)),
        tags=tags,
        first_seen=raw_node.get("first_seen"),
        degree_at_first_seen=int(raw_node.get("degree_at_first_seen", 0)),
        in_value_total=int(raw_node.get("in_value_total", 0)),
        out_value_total=int(raw_node.get("out_value_total", 0)),
        bridge_count=int(raw_node.get("bridge_count", 0)),
        dex_swap_count=int(raw_node.get("dex_swap_count", 0)),
    )


def _parse_edge(raw_edge: Any) -> IREdge:
    if not isinstance(raw_edge, dict):
        raise ValueError("graph edge must be an object")
    source_node = _required_str(raw_edge, "source_node", "graph edge")
    target_node = _required_str(raw_edge, "target_node", "graph edge")
    bridge = str(raw_edge.get("bridge", ""))
    edge_id = str(raw_edge.get("edge_id") or _case_edge_id("graph", 0, source_node, target_node, bridge))
    return IREdge(
        edge_id=edge_id,
        source_node=source_node,
        target_node=target_node,
        edge_type=str(raw_edge.get("edge_type", EdgeType.CROSS_CHAIN_BRIDGE)),
        operator=str(raw_edge.get("operator", TaintOperator.LOCK)),
        bridge=bridge,
        source_event=raw_edge.get("source_event"),
        target_event=raw_edge.get("target_event"),
        value=int(raw_edge.get("value", 0)),
        asset=raw_edge.get("asset"),
        timestamp=raw_edge.get("timestamp", ""),
        block_number=int(raw_edge.get("block_number", 0)),
        fee_bound_pct=float(raw_edge.get("fee_bound_pct", 0.0)),
    )


def _ensure_endpoint_nodes(edge: IREdge, nodes: dict[str, IRNode]) -> None:
    for endpoint in (edge.source_node, edge.target_node):
        if endpoint in nodes:
            continue
        chain, address = endpoint.split(":", 1)
        _ensure_node(nodes, chain, address, frozenset())


def _ensure_node(
    nodes: dict[str, IRNode],
    chain: str,
    address: str,
    tags: frozenset[str],
) -> None:
    key = node_id(chain, address)
    if key in nodes:
        existing = nodes[key]
        nodes[key] = IRNode(
            node_id=existing.node_id,
            address=existing.address,
            chain=existing.chain,
            node_type=existing.node_type,
            tags=existing.tags | tags,
            first_seen=existing.first_seen,
            degree_at_first_seen=existing.degree_at_first_seen,
            in_value_total=existing.in_value_total,
            out_value_total=existing.out_value_total,
            bridge_count=existing.bridge_count,
            dex_swap_count=existing.dex_swap_count,
        )
        return
    nodes[key] = IRNode(
        node_id=key,
        address=address.lower(),
        chain=chain,
        node_type=NodeType.EOA,
        tags=tags,
        first_seen=None,
        degree_at_first_seen=0,
        in_value_total=0,
        out_value_total=0,
        bridge_count=0,
        dex_swap_count=0,
    )


def _load_benign_addresses(manifest: dict[str, Any], path: Path | None) -> list[str]:
    if path is not None:
        return _load_benign_address_file(path)
    raw = manifest.get("benign_addresses", [])
    if isinstance(raw, list):
        return [str(address) for address in raw]
    raise ValueError("benign_addresses must be a list when no benign address file is provided")


def _load_benign_address_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = _load_structured_file(path)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
        if isinstance(loaded, dict) and isinstance(loaded.get("addresses"), list):
            return [str(item) for item in loaded["addresses"]]
        raise ValueError("benign JSON file must be a list or contain an addresses list")
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "address" not in (reader.fieldnames or []):
                raise ValueError("benign CSV file must contain an address column")
            return [row["address"] for row in reader if row.get("address")]
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        if "address" not in frame.columns:
            raise ValueError("benign parquet file must contain an address column")
        return [str(value) for value in frame["address"].dropna().tolist()]
    raise ValueError(f"unsupported benign address format: {path}")


def _fee_bound_pct(bridge: str, hop: dict[str, Any]) -> float:
    if "fee_bound_pct" in hop:
        return float(hop["fee_bound_pct"])
    try:
        return float(Config.load().bridge_spec(bridge).get("fee_bound_pct", 0.0))
    except (AttributeError, FileNotFoundError, KeyError):
        return 0.0


def _case_edge_id(case_id: str, hop_index: int, source_id: str, target_id: str, bridge: str) -> str:
    digest = hashlib.sha256(f"{case_id}:{hop_index}:{source_id}:{target_id}:{bridge}".encode()).hexdigest()
    return f"case_edge_{digest[:16]}"


def _required_str(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"{context} requires {key}")
    return str(value)
