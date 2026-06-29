"""Experiment runner for CrossTaint synthetic and corpus benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from crosstaint.baselines import (
    GraphDegreeBaseline,
    HeuristicBaseline,
    IntraChainCutoffBaseline,
    NaiveBaseline,
)
from crosstaint.config import Config
from crosstaint.eval import (
    compute_f1,
    compute_false_positive_rate,
    compute_hop_recall,
    compute_precision,
    compute_wilcoxon,
)
from crosstaint.experiments.corpus import BenchmarkCase, BenchmarkDataset, load_benchmark_dataset
from crosstaint.experiments.external import ExternalBaseline, load_external_baseline_specs
from crosstaint.indexer import SyntheticGraphBuilder
from crosstaint.pipeline import InferenceStack
from crosstaint.synth import BRIDGE_CHAINS, BRIDGES, BRIDGE_USAGE_WEIGHTS, SyntheticBenignGenerator
from crosstaint.types import IRNode, PropagationResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    num_benign: int = 100
    num_exploit: int = 10
    num_runs: int = 5
    seed: int = 42
    rho: float = 0.81
    threshold: float = 0.04
    output_dir: str = "./results"
    benchmark_manifest: str | None = None
    benign_addresses: str | None = None
    external_baselines: str | None = None
    require_two_source_ground_truth: bool = True


@dataclass(slots=True)
class ExploitCase:
    case_id: str
    origin_address: str
    chain: str
    value: int
    hops: list[tuple[str, str, str]]
    ground_truth_addresses: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MethodResult:
    name: str
    hop_recalls: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    fprs: list[float] = field(default_factory=list)
    f1s: list[float] = field(default_factory=list)
    runtimes_ms: list[float] = field(default_factory=list)

    def add_run(self, recall: float, precision: float, fpr: float, runtime_ms: float) -> None:
        self.hop_recalls.append(recall)
        self.precisions.append(precision)
        self.fprs.append(fpr)
        self.f1s.append(compute_f1(recall, precision))
        self.runtimes_ms.append(runtime_ms)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hop_recall": _stats(self.hop_recalls),
            "precision": _stats(self.precisions),
            "fpr": _stats(self.fprs),
            "f1": _stats(self.f1s),
            "runtime_ms": _stats(self.runtimes_ms),
        }


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def _node_id(chain: str, address: str) -> str:
    return f"{chain}:{address.lower()}"


def _address_from_digest(seed: int, label: str) -> str:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return f"0x{digest[:40]}"


def _sample_bridge_for_chain(rng: np.random.Generator, chain: str) -> str:
    candidates = [
        bridge for bridge in BRIDGES
        if chain in BRIDGE_CHAINS.get(bridge, [])
    ]
    if not candidates:
        candidates = list(BRIDGES)
    weights = np.array([BRIDGE_USAGE_WEIGHTS.get(b, 0.01) for b in candidates], dtype=np.float64)
    weights = weights / weights.sum()
    return str(rng.choice(candidates, p=weights))


def _sample_destination_chain(rng: np.random.Generator, bridge: str, source_chain: str) -> str:
    candidates = [
        chain for chain in BRIDGE_CHAINS.get(bridge, [])
        if chain != source_chain
    ]
    if not candidates:
        candidates = [chain for chain in ("ethereum", "bsc", "polygon", "arbitrum", "avalanche") if chain != source_chain]
    return str(rng.choice(candidates))


def _generate_exploit_cases(n_cases: int, seed: int) -> list[ExploitCase]:
    rng = np.random.default_rng(seed)
    base_chains = ["ethereum", "bsc", "polygon", "arbitrum", "avalanche"]
    cases: list[ExploitCase] = []

    for case_idx in range(n_cases):
        current_chain = str(rng.choice(base_chains))
        origin = _address_from_digest(seed, f"exploit:{case_idx}:origin")
        n_hops = int(rng.integers(2, 6))
        value = int(10 ** rng.uniform(17.0, 20.0))
        hops: list[tuple[str, str, str]] = []

        for _ in range(n_hops):
            bridge = _sample_bridge_for_chain(rng, current_chain)
            dest_chain = _sample_destination_chain(rng, bridge, current_chain)
            hops.append((current_chain, dest_chain, bridge))
            current_chain = dest_chain

        case_id = str(uuid5(NAMESPACE_URL, f"crosstaint:{seed}:{case_idx}:{origin}"))
        cases.append(
            ExploitCase(
                case_id=case_id,
                origin_address=origin,
                chain=hops[0][0],
                value=value,
                hops=hops,
            )
        )

    return cases


def _case_ground_truth(nodes: list[IRNode]) -> list[str]:
    return [node.address for node in nodes[1:]]


def _graph_repr(nodes: dict[str, IRNode], edges: list) -> dict[str, object]:
    return {
        "nodes": [
            {
                "node_id": node.node_id,
                "address": node.address,
                "chain": node.chain,
                "node_type": node.node_type,
                "degree": node.bridge_count + node.dex_swap_count,
                "bridge_count": node.bridge_count,
                "dex_swap_count": node.dex_swap_count,
            }
            for node in nodes.values()
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "source": edge.source_node,
                "target": edge.target_node,
                "edge_type": edge.edge_type,
                "bridge": edge.bridge,
                "value": edge.value,
                "asset": edge.asset,
            }
            for edge in edges
        ],
    }


class _CaseData:
    def __init__(self, origin: str, origin_chain: str, origin_value: int, graph: dict[str, object]) -> None:
        self.origin = origin
        self.origin_chain = origin_chain
        self.origin_value = origin_value
        self.graph = graph


def _run_baseline(baseline, origin_node_id: str, origin_chain: str, origin_value: int, graph: dict[str, object]) -> PropagationResult:
    baseline.fit(None)
    return baseline.predict(_CaseData(origin_node_id, origin_chain, origin_value, graph))


def _predicted_addresses(result: PropagationResult) -> list[str]:
    return [entry.address for entry in result.suspect_set]


def _build_methods(external_baseline_path: str | None) -> dict[str, object]:
    methods: dict[str, object] = {
        "CrossTaint": None,
        "NaiveBaseline": NaiveBaseline(),
        "HeuristicBaseline": HeuristicBaseline(),
        "IntraChainCutoffBaseline": IntraChainCutoffBaseline(),
        "GraphDegreeBaseline": GraphDegreeBaseline(),
    }
    for spec in load_external_baseline_specs(Path(external_baseline_path) if external_baseline_path else None):
        if spec.name in methods:
            raise ValueError(f"duplicate method name: {spec.name}")
        methods[spec.name] = ExternalBaseline(spec)
    return methods


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    Config.reset()
    Config.load()

    start_time = time.monotonic()
    methods = _build_methods(cfg.external_baselines)
    all_results = {name: MethodResult(name) for name in methods}
    per_case: list[dict[str, object]] = []

    for run_idx in range(cfg.num_runs):
        seed = cfg.seed + run_idx
        logger.info("Run %s/%s with seed %s", run_idx + 1, cfg.num_runs, seed)

        generator = SyntheticBenignGenerator(seed=seed)
        benign_trajs = generator.generate(num_trajectories=cfg.num_benign)
        exploit_cases = _generate_exploit_cases(cfg.num_exploit, seed)

        builder = SyntheticGraphBuilder(
            benign_trajs,
            exploit_addresses=[case.origin_address for case in exploit_cases],
        )
        nodes, _ = builder.build()

        for case in exploit_cases:
            case_nodes, _case_edges = builder.create_exploit_case(
                origin=case.origin_address,
                chain=case.chain,
                value=case.value,
                hops=case.hops,
            )
            case.ground_truth_addresses = _case_ground_truth(case_nodes)

        nodes, edge_map = builder.snapshot()
        edges = list(edge_map.values())
        graph = _graph_repr(nodes, edges)
        benign_addresses = [
            address for trajectory in benign_trajs for address in trajectory.addresses
        ]

        for case in exploit_cases:
            origin_node_id = _node_id(case.chain, case.origin_address)
            stack = InferenceStack.from_config(
                rho=cfg.rho,
                threshold=cfg.threshold,
                seed=seed,
            )
            ct_result = stack.propagate(
                origin=origin_node_id,
                origin_chain=case.chain,
                origin_value=case.value,
                graph=nodes,
                edges=edges,
                case_id=case.case_id,
            )
            ct_predicted = _predicted_addresses(ct_result)
            ct_recall = compute_hop_recall([ct_predicted], [case.ground_truth_addresses])
            ct_precision = compute_precision([ct_predicted], [case.ground_truth_addresses])
            ct_fpr = compute_false_positive_rate(ct_predicted, benign_addresses)
            all_results["CrossTaint"].add_run(ct_recall, ct_precision, ct_fpr, float(ct_result.runtime_ms))

            per_case.append(
                {
                    "run": run_idx,
                    "case_id": case.case_id,
                    "method": "CrossTaint",
                    "hop_recall": ct_recall,
                    "precision": ct_precision,
                    "fpr": ct_fpr,
                    "runtime_ms": ct_result.runtime_ms,
                    "ground_truth_hops": len(case.ground_truth_addresses),
                    "predicted_count": len(ct_predicted),
                }
            )

            for baseline_name, baseline in methods.items():
                if baseline_name == "CrossTaint":
                    continue
                result = _run_baseline(
                    baseline,
                    origin_node_id,
                    case.chain,
                    case.value,
                    graph,
                )
                predicted = _predicted_addresses(result)
                recall = compute_hop_recall([predicted], [case.ground_truth_addresses])
                precision = compute_precision([predicted], [case.ground_truth_addresses])
                fpr = compute_false_positive_rate(predicted, benign_addresses)
                all_results[baseline_name].add_run(recall, precision, fpr, float(result.runtime_ms))
                per_case.append(
                    {
                        "run": run_idx,
                        "case_id": case.case_id,
                        "method": baseline_name,
                        "hop_recall": recall,
                        "precision": precision,
                        "fpr": fpr,
                        "runtime_ms": result.runtime_ms,
                        "ground_truth_hops": len(case.ground_truth_addresses),
                        "predicted_count": len(predicted),
                    }
                )

    summary: dict[str, object] = {name: result.summary() for name, result in all_results.items()}
    _attach_significance(summary, all_results)
    scope = "synthetic_benchmark"
    summary["_metadata"] = _metadata(cfg, time.monotonic() - start_time, scope, "")
    summary["_per_case"] = per_case
    return summary


def run_corpus_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    if not cfg.benchmark_manifest:
        raise ValueError("benchmark_manifest is required for corpus benchmark mode")

    Config.reset()
    Config.load()
    dataset = load_benchmark_dataset(
        manifest_path=Path(cfg.benchmark_manifest),
        benign_addresses_path=Path(cfg.benign_addresses) if cfg.benign_addresses else None,
        require_two_source_ground_truth=cfg.require_two_source_ground_truth,
    )

    start_time = time.monotonic()
    methods = _build_methods(cfg.external_baselines)
    all_results = {name: MethodResult(name) for name in methods}
    per_case: list[dict[str, object]] = []
    graph = _graph_repr(dataset.nodes, dataset.edges)

    for run_idx in range(cfg.num_runs):
        seed = cfg.seed + run_idx
        logger.info("Corpus run %s/%s with seed %s", run_idx + 1, cfg.num_runs, seed)
        for case in dataset.cases:
            stack = InferenceStack.from_config(
                rho=cfg.rho,
                threshold=cfg.threshold,
                seed=seed,
            )
            ct_result = stack.propagate(
                origin=case.origin_node_id,
                origin_chain=case.origin_chain,
                origin_value=case.origin_value,
                graph=dataset.nodes,
                edges=dataset.edges,
                case_id=case.case_id,
            )
            _record_method_result(
                method_name="CrossTaint",
                result=ct_result,
                ground_truth=case.ground_truth_addresses,
                benign_addresses=dataset.benign_addresses,
                run_idx=run_idx,
                case=case,
                all_results=all_results,
                per_case=per_case,
            )

            for method_name, method in methods.items():
                if method_name == "CrossTaint":
                    continue
                result = _run_baseline(
                    method,
                    case.origin_node_id,
                    case.origin_chain,
                    case.origin_value,
                    graph,
                )
                _record_method_result(
                    method_name=method_name,
                    result=result,
                    ground_truth=case.ground_truth_addresses,
                    benign_addresses=dataset.benign_addresses,
                    run_idx=run_idx,
                    case=case,
                    all_results=all_results,
                    per_case=per_case,
                )

    summary: dict[str, object] = {name: result.summary() for name, result in all_results.items()}
    _attach_significance(summary, all_results)
    summary["_metadata"] = _metadata(cfg, time.monotonic() - start_time, "corpus_benchmark", dataset.manifest_hash)
    summary["_dataset"] = {
        "num_cases": len(dataset.cases),
        "num_nodes": len(dataset.nodes),
        "num_edges": len(dataset.edges),
        "num_benign_addresses": len(dataset.benign_addresses),
        **dataset.metadata,
    }
    summary["_per_case"] = per_case
    return summary


def _record_method_result(
    method_name: str,
    result: PropagationResult,
    ground_truth: list[str],
    benign_addresses: list[str],
    run_idx: int,
    case: BenchmarkCase,
    all_results: dict[str, MethodResult],
    per_case: list[dict[str, object]],
) -> None:
    predicted = _predicted_addresses(result)
    recall = compute_hop_recall([predicted], [ground_truth])
    precision = compute_precision([predicted], [ground_truth])
    fpr = compute_false_positive_rate(predicted, benign_addresses)
    all_results[method_name].add_run(recall, precision, fpr, float(result.runtime_ms))
    per_case.append(
        {
            "run": run_idx,
            "case_id": case.case_id,
            "method": method_name,
            "hop_recall": recall,
            "precision": precision,
            "fpr": fpr,
            "runtime_ms": result.runtime_ms,
            "ground_truth_hops": len(ground_truth),
            "predicted_count": len(predicted),
        }
    )


def _attach_significance(summary: dict[str, object], results: dict[str, MethodResult]) -> None:
    ct = results["CrossTaint"].hop_recalls
    baselines = [result for name, result in results.items() if name != "CrossTaint"]
    if not ct or not baselines:
        return
    best_baseline = [
        max(result.hop_recalls[idx] for result in baselines if idx < len(result.hop_recalls))
        for idx in range(len(ct))
    ]
    wilcoxon = compute_wilcoxon(ct, best_baseline, alternative="greater")
    gap_ci = _bootstrap_gap_ci(ct, best_baseline)
    crosstaint_summary = summary.get("CrossTaint")
    if isinstance(crosstaint_summary, dict):
        crosstaint_summary["wilcoxon_vs_best_baseline"] = {
            "statistic": float(wilcoxon.statistic),
            "p_value": float(wilcoxon.p_value),
            "significant": bool(wilcoxon.significant),
            "n_pairs": int(wilcoxon.n_pairs),
        }
        crosstaint_summary["gap_vs_best_baseline"] = gap_ci


def _bootstrap_gap_ci(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | int]:
    if not values_a or not values_b:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    n = min(len(values_a), len(values_b))
    gaps = np.array(values_a[:n], dtype=np.float64) - np.array(values_b[:n], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(gaps[idx])))
    return {
        "mean": float(np.mean(gaps)),
        "ci_lower": float(np.percentile(samples, alpha / 2 * 100)),
        "ci_upper": float(np.percentile(samples, (1 - alpha / 2) * 100)),
        "n": n,
        "n_resamples": n_resamples,
    }


def _metadata(
    cfg: ExperimentConfig,
    elapsed_seconds: float,
    scope: str,
    manifest_hash: str,
) -> dict[str, object]:
    return {
        "config": _public_config(cfg),
        "elapsed_seconds": elapsed_seconds,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "config_hash": Config.config_hash(),
        "scope": scope,
        "manifest_hash": manifest_hash,
        "paper_evidence_level": _paper_evidence_level(scope),
    }


def _public_config(cfg: ExperimentConfig) -> dict[str, object]:
    data = asdict(cfg)
    for key in ("output_dir", "benchmark_manifest", "benign_addresses", "external_baselines"):
        value = data.get(key)
        if value:
            data[key] = Path(str(value)).name
    return data


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"numpy": np.__version__}
    try:
        import pandas as pd
        versions["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    return versions


def _paper_evidence_level(scope: str) -> str:
    if scope == "corpus_benchmark":
        return "corpus_benchmark"
    return "synthetic_benchmark"


def write_results(results: dict[str, object], output_dir: str, filename: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / filename
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results_file


def print_results(results: dict[str, object]) -> None:
    logger.info("Experiment Results Summary")
    for name, data in results.items():
        if name.startswith("_") or not isinstance(data, dict):
            continue
        logger.info("%s", name)
        for metric in ("hop_recall", "precision", "fpr", "f1", "runtime_ms"):
            metric_data = data.get(metric)
            if not isinstance(metric_data, dict) or metric_data.get("n", 0) == 0:
                continue
            logger.info(
                "  %-12s mean=%.4f std=%.4f n=%s",
                metric,
                metric_data.get("mean", 0.0),
                metric_data.get("std", 0.0),
                metric_data.get("n", 0),
            )
        wilcoxon = data.get("wilcoxon_vs_best_baseline")
        if isinstance(wilcoxon, dict):
            logger.info(
                "  Wilcoxon p=%.4f significant=%s n=%s",
                wilcoxon["p_value"],
                wilcoxon["significant"],
                wilcoxon["n_pairs"],
            )


def run_local_benchmark(cfg: ExperimentConfig) -> Path:
    logger.info(
        "Running local benchmark: benign=%s exploit=%s runs=%s",
        cfg.num_benign,
        cfg.num_exploit,
        cfg.num_runs,
    )
    results = run_experiment(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = write_results(results, cfg.output_dir, f"results_{timestamp}.json")
    logger.info("Results saved to %s", results_file)
    print_results(results)
    return results_file


def run_corpus_benchmark(cfg: ExperimentConfig) -> Path:
    logger.info("Running corpus benchmark from %s", cfg.benchmark_manifest)
    results = run_corpus_experiment(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = write_results(results, cfg.output_dir, f"corpus_results_{timestamp}.json")
    logger.info("Results saved to %s", results_file)
    print_results(results)
    return results_file


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="CrossTaint experiment runner")
    parser.add_argument("--local-benchmark", action="store_true", help="Run local synthetic benchmark")
    parser.add_argument("--full-benchmark", action="store_true", help="Run corpus benchmark from a manifest")
    parser.add_argument("--num-benign", type=int, default=100)
    parser.add_argument("--num-exploit", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.81)
    parser.add_argument("--threshold", type=float, default=0.04)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--benchmark-manifest", type=str, default=None)
    parser.add_argument("--benign-addresses", type=str, default=None)
    parser.add_argument("--external-baselines", type=str, default=None)
    parser.add_argument(
        "--allow-single-source-ground-truth",
        action="store_true",
        help="Disable two-source validation for local debugging only",
    )

    args = parser.parse_args()

    cfg = ExperimentConfig(
        num_benign=args.num_benign,
        num_exploit=args.num_exploit,
        num_runs=args.num_runs,
        seed=args.seed,
        rho=args.rho,
        threshold=args.threshold,
        output_dir=args.output_dir,
        benchmark_manifest=args.benchmark_manifest,
        benign_addresses=args.benign_addresses,
        external_baselines=args.external_baselines,
        require_two_source_ground_truth=not args.allow_single_source_ground_truth,
    )
    if args.full_benchmark or args.benchmark_manifest:
        run_corpus_benchmark(cfg)
    elif args.local_benchmark:
        run_local_benchmark(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
