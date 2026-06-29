"""End-to-end inference stack for CrossTaint propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from crosstaint.config import Config
from crosstaint.matcher.calibration import PlattCalibrator
from crosstaint.matcher.featurizer import EventFeaturizer
from crosstaint.matcher.model import BridgeEventMatcher
from crosstaint.propagation import PropagationEngine
from crosstaint.pseudonym.model import HGTResolver, HeteroGraph, HeteroNode
from crosstaint.types import (
    DecodedEvent,
    EdgeType,
    IREdge,
    IRNode,
    NodeFeatures,
    PropagationResult,
    SimilarityOutput,
)


class EdgeMatcher:
    """Bridge-event matcher adapter used by the propagation engine."""

    def __init__(
        self,
        model: BridgeEventMatcher,
        featurizer: EventFeaturizer,
        event_index: dict[str, DecodedEvent],
        calibrator: PlattCalibrator | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.featurizer = featurizer
        self.event_index = event_index
        self.calibrator = calibrator
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        event_index: dict[str, DecodedEvent],
        config: Config | None = None,
        device: str = "cpu",
    ) -> "EdgeMatcher":
        cfg = config or Config.load()
        matcher_cfg = cfg.matcher
        model = BridgeEventMatcher(
            d_model=int(matcher_cfg.get("embedding_dim", 128)),
            n_heads=int(matcher_cfg.get("num_heads", 4)),
            n_layers=int(matcher_cfg.get("num_layers", 4)),
            dim_feedforward=int(matcher_cfg.get("ff_dim", 512)),
            dropout=float(matcher_cfg.get("dropout", 0.1)),
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

        calibrator = None

        return cls(
            model=model,
            featurizer=EventFeaturizer(seed=int(cfg.propagation.get("seed", 42))),
            event_index=event_index,
            calibrator=calibrator,
            device=device,
        )

    def score_edge(self, edge: IREdge) -> float:
        if edge.edge_type != EdgeType.CROSS_CHAIN_BRIDGE:
            return 1.0
        if not edge.source_event or not edge.target_event:
            return 1.0

        source = self.event_index.get(edge.source_event)
        dest = self.event_index.get(edge.target_event)
        if source is None or dest is None:
            return 1.0

        source_tensor = self.featurizer.event_to_tensor(self.featurizer.featurize(source))
        dest_tensor = self.featurizer.event_to_tensor(self.featurizer.featurize(dest))
        with torch.no_grad():
            output = self.model(
                source_tensor.to(self.device),
                dest_tensor.to(self.device),
            )
            score = float(output.score.squeeze().item())

        if self.calibrator is not None:
            score = float(self.calibrator.calibrate(score))
        return max(0.0, min(1.0, score))


class PseudonymResolver:
    """HGT resolver adapter used by the propagation engine."""

    def __init__(
        self,
        model: HGTResolver,
        nodes: dict[str, IRNode],
        edges: list[IREdge],
        cold_start_degree: int = 2,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.nodes = nodes
        self.edges = edges
        self.cold_start_degree = cold_start_degree
        self.device = device
        self._graph = self._build_graph(nodes, edges)
        self._embeddings: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        nodes: dict[str, IRNode],
        edges: list[IREdge],
        config: Config | None = None,
        device: str = "cpu",
    ) -> "PseudonymResolver":
        cfg = config or Config.load()
        pseudonym_cfg = cfg.pseudonym
        model = HGTResolver(
            n_layers=int(pseudonym_cfg.get("num_layers", 3)),
            hidden_dim=int(pseudonym_cfg.get("hidden_dim", 256)),
            n_heads=int(pseudonym_cfg.get("num_heads", 4)),
            dropout=float(pseudonym_cfg.get("dropout", 0.1)),
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

        return cls(
            model=model,
            nodes=nodes,
            edges=edges,
            cold_start_degree=int(pseudonym_cfg.get("cold_start_threshold", 2)),
            device=device,
        )

    def _build_graph(
        self,
        nodes: dict[str, IRNode],
        edges: list[IREdge],
    ) -> HeteroGraph:
        hetero_nodes = [
            HeteroNode(
                node_id=node_id,
                chain=node.chain,
                features=torch.tensor(
                    NodeFeatures(
                        in_value=node.in_value_total,
                        out_value=node.out_value_total,
                        bridge_count=node.bridge_count,
                        dex_swap_count=node.dex_swap_count,
                        degree=node.bridge_count + node.dex_swap_count,
                        degree_at_first_seen=node.degree_at_first_seen,
                        chain=node.chain,
                        node_type=node.node_type,
                    ).to_vector(),
                    dtype=torch.float32,
                ),
            )
            for node_id, node in nodes.items()
        ]
        edge_map: dict[str, list[tuple[str, str]]] = {
            EdgeType.INTRA_CHAIN_TRANSFER: [],
            EdgeType.CROSS_CHAIN_BRIDGE: [],
            EdgeType.DEX_SWAP: [],
        }
        for edge in edges:
            if edge.edge_type in edge_map:
                edge_map[edge.edge_type].append((edge.source_node, edge.target_node))
        return HeteroGraph(nodes={"global": hetero_nodes}, edges=edge_map)

    def _ensure_embeddings(self) -> dict[str, torch.Tensor]:
        if self._embeddings is None:
            with torch.no_grad():
                self._embeddings = self.model.forward(self._graph)
        return self._embeddings

    def resolve(self, source: str, target: str) -> SimilarityOutput:
        target_node = self.nodes.get(target)
        if target_node is None:
            return SimilarityOutput(source, target, 1.0, False)
        if target_node.degree_at_first_seen <= self.cold_start_degree:
            return SimilarityOutput(source, target, 1.0, True)

        embeddings = self._ensure_embeddings()
        return self.model.resolve(source, target, embeddings, self._graph)


class InferenceStack:
    """Configured CrossTaint inference pipeline."""

    def __init__(
        self,
        propagation: PropagationEngine,
        matcher: EdgeMatcher | None = None,
        resolver: PseudonymResolver | None = None,
    ) -> None:
        self.propagation = propagation
        self.matcher = matcher
        self.resolver = resolver

    @classmethod
    def from_config(
        cls,
        config_dir: Path | None = None,
        matcher_checkpoint: Path | None = None,
        resolver_checkpoint: Path | None = None,
        event_index: dict[str, DecodedEvent] | None = None,
        nodes: dict[str, IRNode] | None = None,
        edges: list[IREdge] | None = None,
        device: str = "cpu",
        **propagation_overrides: Any,
    ) -> "InferenceStack":
        Config.reset()
        config = Config.load(config_dir)
        propagation = PropagationEngine.from_config(config, **propagation_overrides)

        matcher = None
        if matcher_checkpoint is not None and event_index is not None:
            matcher = EdgeMatcher.from_checkpoint(
                matcher_checkpoint,
                event_index,
                config=config,
                device=device,
            )

        resolver = None
        if resolver_checkpoint is not None and nodes is not None and edges is not None:
            resolver = PseudonymResolver.from_checkpoint(
                resolver_checkpoint,
                nodes,
                edges,
                config=config,
                device=device,
            )

        return cls(propagation=propagation, matcher=matcher, resolver=resolver)

    def propagate(
        self,
        origin: str,
        origin_chain: str,
        origin_value: int,
        graph: dict[str, IRNode],
        edges: list[IREdge],
        case_id: str,
    ) -> PropagationResult:
        if self.resolver is not None:
            self.resolver.nodes = graph
            self.resolver.edges = edges
            self.resolver._graph = self.resolver._build_graph(graph, edges)
            self.resolver._embeddings = None

        return self.propagation.propagate(
            origin=origin,
            origin_chain=origin_chain,
            origin_value=origin_value,
            graph=graph,
            edges=edges,
            case_id=case_id,
            matcher=self.matcher,
            resolver=self.resolver,
        )