"""CrossTaint experiment runners and corpus loaders.

The public objects are imported lazily so that manifest inspection does not
initialise the PyTorch model stack.

Modified in derived release v1.1.0 to introduce the lazy public imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "load_benchmark_dataset",
    "ExternalBaseline",
    "ExternalBaselineSpec",
    "load_external_baseline_specs",
    "ExperimentConfig",
    "ExploitCase",
    "MethodResult",
    "run_experiment",
    "run_local_benchmark",
    "run_corpus_benchmark",
    "write_results",
    "print_results",
]


_EXPORT_MODULES = {
    "BenchmarkCase": "crosstaint.experiments.corpus",
    "BenchmarkDataset": "crosstaint.experiments.corpus",
    "load_benchmark_dataset": "crosstaint.experiments.corpus",
    "ExternalBaseline": "crosstaint.experiments.external",
    "ExternalBaselineSpec": "crosstaint.experiments.external",
    "load_external_baseline_specs": "crosstaint.experiments.external",
    "ExperimentConfig": "crosstaint.experiments.run",
    "ExploitCase": "crosstaint.experiments.run",
    "MethodResult": "crosstaint.experiments.run",
    "run_experiment": "crosstaint.experiments.run",
    "run_local_benchmark": "crosstaint.experiments.run",
    "run_corpus_benchmark": "crosstaint.experiments.run",
    "write_results": "crosstaint.experiments.run",
    "print_results": "crosstaint.experiments.run",
}


def __getattr__(name: str) -> Any:
    """Load public experiment objects only when they are requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
