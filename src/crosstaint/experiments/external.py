"""External baseline execution contract."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml

from crosstaint.types import PropagationResult, SuspectEntry


@dataclass(frozen=True, slots=True)
class ExternalBaselineSpec:
    name: str
    command: list[str]
    timeout_seconds: int = 3600


class ExternalBaseline:
    def __init__(self, spec: ExternalBaselineSpec) -> None:
        if not spec.command:
            raise ValueError(f"external baseline {spec.name} command cannot be empty")
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def fit(self, train_data: Any) -> None:
        del train_data

    def predict(self, case_data: Any) -> PropagationResult:
        start_time = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"crosstaint_{self.spec.name}_") as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "case.json"
            output_path = tmp_path / "result.json"
            input_payload = _case_payload(case_data)
            with open(input_path, "w", encoding="utf-8") as handle:
                json.dump(input_payload, handle)

            env = os.environ.copy()
            env["CROSSTAINT_CASE_JSON"] = str(input_path)
            env["CROSSTAINT_RESULT_JSON"] = str(output_path)
            env["CROSSTAINT_BASELINE_NAME"] = self.spec.name

            completed = subprocess.run(
                self.spec.command,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.spec.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"external baseline {self.spec.name} failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            if not output_path.exists():
                raise RuntimeError(f"external baseline {self.spec.name} did not write result JSON")

            with open(output_path, "r", encoding="utf-8") as handle:
                output = json.load(handle)

        runtime_ms = float(output.get("runtime_ms", (time.perf_counter() - start_time) * 1000))
        suspect_set = tuple(_parse_suspect(entry, getattr(case_data, "origin")) for entry in output.get("suspect_set", []))
        case_id = str(output.get("case_id") or uuid5(NAMESPACE_URL, f"{self.spec.name}:{getattr(case_data, 'origin')}"))
        return PropagationResult(
            case_id=case_id,
            origin=getattr(case_data, "origin"),
            origin_chain=getattr(case_data, "origin_chain"),
            origin_value=int(getattr(case_data, "origin_value")),
            suspect_set=suspect_set,
            runtime_ms=runtime_ms,
            config_hash=f"external:{self.spec.name}",
            seed=0,
            path_count=len(suspect_set),
        )


def load_external_baseline_specs(path: Path | None) -> list[ExternalBaselineSpec]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    if isinstance(raw, dict):
        raw_specs = raw.get("baselines", [])
    else:
        raw_specs = raw
    if not isinstance(raw_specs, list):
        raise ValueError("external baseline spec must be a list or contain a baselines list")

    specs: list[ExternalBaselineSpec] = []
    for idx, item in enumerate(raw_specs):
        if not isinstance(item, dict):
            raise ValueError(f"external baseline entry {idx} must be an object")
        name = str(item.get("name", "")).strip()
        command = item.get("command", [])
        if not name:
            raise ValueError(f"external baseline entry {idx} requires name")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            raise ValueError(f"external baseline {name} command must be a list of strings")
        specs.append(
            ExternalBaselineSpec(
                name=name,
                command=command,
                timeout_seconds=int(item.get("timeout_seconds", 3600)),
            )
        )
    return specs


def _case_payload(case_data: Any) -> dict[str, Any]:
    return {
        "origin": getattr(case_data, "origin"),
        "origin_chain": getattr(case_data, "origin_chain"),
        "origin_value": int(getattr(case_data, "origin_value")),
        "graph": getattr(case_data, "graph"),
    }


def _parse_suspect(entry: Any, origin: str) -> SuspectEntry:
    if not isinstance(entry, dict):
        raise ValueError("external baseline suspect entry must be an object")
    address = str(entry.get("address", "")).strip()
    chain = str(entry.get("chain", "")).strip()
    if not address or not chain:
        raise ValueError("external baseline suspect entry requires address and chain")
    flags = entry.get("flags", [])
    if flags is None:
        flags = []
    return SuspectEntry(
        address=address,
        chain=chain,
        taint_score=float(entry.get("taint_score", entry.get("score", 0.0))),
        hop_distance=int(entry.get("hop_distance", 0)),
        predecessors=(str(entry.get("predecessor", origin)),),
        is_cold_start=bool(entry.get("is_cold_start", False)),
        flags=frozenset(str(flag) for flag in flags),
    )
