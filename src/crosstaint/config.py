"""CrossTaint configuration loading.

Modified in derived release v1.1.0 to package defaults and fail clearly when
an explicitly selected configuration directory is invalid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


_ENV_INTERP_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _candidate_config_dirs() -> list[Path]:
    configured = os.environ.get("CROSSTAINT_CONFIG_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        Path.cwd() / "config",
        Path(__file__).resolve().parents[2] / "config",
        Path(__file__).resolve().parent / "default_config",
    ])
    return candidates


def _default_config_dir() -> Path:
    for candidate in _candidate_config_dirs():
        if candidate.exists() and any(candidate.glob("*.yaml")):
            return candidate
    raise FileNotFoundError(
        "no CrossTaint config directory found; set CROSSTAINT_CONFIG_DIR "
        "or run from the framework root"
    )


def _env_interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            env_var = m.group(1)
            return os.environ.get(env_var, m.group(0))
        return _ENV_INTERP_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _env_interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_env_interpolate(v) for v in value]
    return value


def _index_by_id(items: Any, id_key: str) -> dict[str, dict]:
    if isinstance(items, list):
        return {
            str(item[id_key]): item
            for item in items
            if isinstance(item, dict) and id_key in item
        }
    if isinstance(items, dict):
        first_val = next(iter(items.values()), None)
        if isinstance(first_val, dict) and id_key in first_val:
            return {str(item[id_key]): item for item in items.values()}
        return {str(key): value for key, value in items.items() if isinstance(value, dict)}
    return {}


class Config:
    _instance: "Config | None" = None
    _config_hash: str = ""

    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    @classmethod
    def load(cls, config_dir: Path | None = None) -> "Config":
        if cls._instance is not None:
            return cls._instance

        cfg = cls()
        directory = Path(config_dir) if config_dir is not None else _default_config_dir()
        if not directory.is_dir():
            raise FileNotFoundError(f"CrossTaint config directory not found: {directory}")

        yaml_files = sorted(directory.glob("*.yaml"))
        if not yaml_files:
            raise FileNotFoundError(
                f"CrossTaint config directory contains no YAML files: {directory}"
            )
        cfg._loaded = {}

        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            if raw is None:
                raw = {}
            name = yaml_file.stem
            if name in raw:
                cfg._loaded[name] = _env_interpolate(raw[name])
            else:
                cfg._loaded[name] = _env_interpolate(raw)

        raw_str = json.dumps(cfg._loaded, sort_keys=True, default=str)
        cfg._config_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:12]

        cls._instance = cfg
        cls._config_hash = cfg._config_hash
        return cfg

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        cls._config_hash = ""

    @classmethod
    def config_hash(cls) -> str:
        return cls._config_hash

    def __getattr__(self, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        raise AttributeError(
            f"No config section '{name}'. Available: {list(self._loaded.keys())}"
        )

    def get(self, section: str, default: Any = None) -> Any:
        return self._loaded.get(section, default)

    @property
    def chains(self) -> dict[str, dict]:
        return _index_by_id(self._loaded.get("chains"), "chain_id")

    @property
    def bridges(self) -> dict[str, dict]:
        return _index_by_id(self._loaded.get("bridges"), "bridge_id")

    @property
    def matcher(self) -> dict:
        return self._loaded.get("matcher", {})

    @property
    def pseudonym(self) -> dict:
        return self._loaded.get("pseudonym", {})

    @property
    def propagation(self) -> dict:
        return self._loaded.get("propagation", {})

    @property
    def eval(self) -> dict:
        return self._loaded.get("eval", {})

    @property
    def chain_list(self) -> list[str]:
        return list(self.chains.keys())

    @property
    def bridge_list(self) -> list[str]:
        return list(self.bridges.keys())

    def bridge_spec(self, bridge_id: str) -> dict:
        if bridge_id in self.bridges:
            return self.bridges[bridge_id]
        raise KeyError(f"Bridge '{bridge_id}' not found in config")

    def chain_spec(self, chain_id: str) -> dict:
        if chain_id in self.chains:
            return self.chains[chain_id]
        raise KeyError(f"Chain '{chain_id}' not found in config")

    def bridge_fee_bound(self, bridge_id: str, default: float = 0.01) -> float:
        return float(self.bridge_spec(bridge_id).get("fee_bound_pct", default))


def get_config() -> Config:
    return Config.load()
