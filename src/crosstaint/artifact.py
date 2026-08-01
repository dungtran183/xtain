"""Repository-level validation for the public CrossTaint artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


EXPECTED_NAME = "CrossTaint"
EXPECTED_PACKAGE = "crosstaint"
EXPECTED_VERSION = "1.1.0"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_REPOSITORY = "https://github.com/dungtran183/xtain"
EXPECTED_RELEASE = "https://github.com/dungtran183/xtain/tree/v1.1.0"
EXPECTED_RELEASE_STATUS = "released"
EXPECTED_RELEASE_DATE = "2026-08-01"
UPSTREAM_REPOSITORY = "https://github.com/fuondai/xtain"
UPSTREAM_RELEASE = "https://github.com/fuondai/xtain/tree/v1.0.0"
UPSTREAM_COMMIT = "320883a61888ba7a92a6ad738d951a7dcc7ffd23"
UPSTREAM_DOI = "10.5281/zenodo.21021751"
UPSTREAM_ARCHIVE = f"https://doi.org/{UPSTREAM_DOI}"
UPSTREAM_LANDING_METADATA_LICENSE = "CC-BY-4.0"
UPSTREAM_SOURCE_LICENSE = "Apache-2.0"
OFFICIAL_LICENSE_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)


def _find_repository_root(start: Path) -> Path:
    """Find the nearest checkout containing the artifact manifest."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "artifact" / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "could not find a CrossTaint checkout; pass --root PATH from an installed command"
    )


def _load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing file: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path}: {exc}")
    return None


def _project_license(project: dict[str, Any]) -> str:
    license_value = project.get("license", "")
    if isinstance(license_value, dict):
        return str(license_value.get("text", ""))
    return str(license_value)


def _cff_top_level_scalars(path: Path, errors: list[str]) -> dict[str, str]:
    """Read the simple top-level scalars needed for an offline CFF check."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append(f"missing file: {path}")
        return {}
    values: dict[str, str] = {}
    for line in lines:
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip().strip('"').strip("'")
        if value:
            values[key] = value
    return values


def _check_required_paths(
    root: Path,
    required_paths: Any,
    errors: list[str],
) -> int:
    if not isinstance(required_paths, list) or not required_paths:
        errors.append("manifest required_paths must be a non-empty list")
        return 0

    checked = 0
    root_resolved = root.resolve()
    for value in required_paths:
        if not isinstance(value, str) or not value:
            errors.append("manifest required_paths entries must be non-empty strings")
            continue
        path = (root / value).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError:
            errors.append(f"required path escapes repository root: {value}")
            continue
        checked += 1
        if not path.exists():
            errors.append(f"required path does not exist: {value}")
    return checked


def _check_smoke_benchmark(path: Path, errors: list[str]) -> None:
    data = _load_json(path, errors)
    if not isinstance(data, dict):
        return
    if data.get("evidence_scope") != "synthetic_oracle":
        errors.append("smoke benchmark evidence_scope must be 'synthetic_oracle'")
    oracle_metadata = data.get("oracle_metadata")
    if not isinstance(oracle_metadata, dict):
        errors.append("smoke benchmark must contain oracle_metadata")
    elif oracle_metadata.get("independent_provenance") is not False:
        errors.append(
            "smoke benchmark oracle metadata must declare independent_provenance=false"
        )
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("smoke benchmark must contain at least one case")
        return
    for case_index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"smoke benchmark case {case_index} must be an object")
            continue
        if case.get("ground_truth_sources"):
            errors.append(
                f"smoke benchmark case {case_index} must not claim independent ground-truth sources"
            )
        origin = case.get("origin")
        if not isinstance(origin, dict) or not all(
            origin.get(field) not in (None, "") for field in ("address", "chain", "value")
        ):
            errors.append(
                f"smoke benchmark case {case_index} needs origin address, chain, and value"
            )
        hops = case.get("hops")
        if not isinstance(hops, list) or not hops:
            errors.append(f"smoke benchmark case {case_index} needs at least one hop")
            continue
        for hop_index, hop in enumerate(hops):
            if not isinstance(hop, dict):
                errors.append(
                    f"smoke benchmark case {case_index} hop {hop_index} must be an object"
                )
                continue
            for field in ("to_chain", "bridge", "address"):
                if hop.get(field) in (None, ""):
                    errors.append(
                        f"smoke benchmark case {case_index} hop {hop_index} needs {field}"
                    )
            if hop.get("sources"):
                errors.append(
                    f"smoke benchmark case {case_index} hop {hop_index} must not claim independent sources"
                )


def run_synthetic_oracle_smoke(root: Path) -> dict[str, Any]:
    """Exercise fixed-weight propagation over the bundled synthetic oracle."""
    from crosstaint.config import Config
    from crosstaint.experiments.corpus import load_benchmark_dataset
    from crosstaint.propagation.engine import PropagationEngine

    root = root.resolve()
    Config.reset()
    try:
        dataset = load_benchmark_dataset(
            root / "examples" / "smoke-benchmark.json",
            require_two_source_ground_truth=False,
        )
        oracle_metadata = dataset.metadata.get("oracle_metadata", {})
        if dataset.metadata.get("evidence_scope") != "synthetic_oracle" or (
            not isinstance(oracle_metadata, dict)
            or oracle_metadata.get("independent_provenance") is not False
        ):
            raise ValueError(
                "bundled smoke fixture must identify synthetic-oracle metadata "
                "with independent_provenance=false"
            )
        if len(dataset.cases) != 1:
            raise ValueError("bundled smoke fixture must contain exactly one case")

        config = Config.load(root / "config")
        case = dataset.cases[0]
        result = PropagationEngine.from_config(config).propagate(
            origin=case.origin_node_id,
            origin_chain=case.origin_chain,
            origin_value=case.origin_value,
            graph=dataset.nodes,
            edges=dataset.edges,
            case_id=case.case_id,
        )
        recovered = [entry.address for entry in result.suspect_set]
        if recovered != case.ground_truth_addresses:
            raise AssertionError(
                "synthetic-oracle smoke mismatch: "
                f"expected {case.ground_truth_addresses!r}, got {recovered!r}"
            )
        return {
            "ok": True,
            "fixture": "examples/smoke-benchmark.json",
            "evidence_scope": "synthetic_oracle",
            "independent_provenance": False,
            "cases": 1,
            "hops_recovered": len(recovered),
            "config_hash": result.config_hash,
        }
    finally:
        Config.reset()


def validate_release(root: Path) -> dict[str, Any]:
    """Validate release metadata and repository structure without network access."""
    root = root.resolve()
    errors: list[str] = []

    manifest_path = root / "artifact" / "manifest.json"
    schema_path = root / "artifact" / "manifest.schema.json"
    manifest = _load_json(manifest_path, errors)
    schema = _load_json(schema_path, errors)

    required_paths_checked = 0
    if isinstance(manifest, dict):
        artifact = manifest.get("artifact")
        if not isinstance(artifact, dict):
            errors.append("manifest artifact field must be an object")
        else:
            expected = {
                "name": EXPECTED_NAME,
                "package": EXPECTED_PACKAGE,
                "version": EXPECTED_VERSION,
                "license": EXPECTED_LICENSE,
                "repository": EXPECTED_REPOSITORY,
                "release": EXPECTED_RELEASE,
                "release_status": EXPECTED_RELEASE_STATUS,
                "release_date": EXPECTED_RELEASE_DATE,
            }
            for key, value in expected.items():
                if artifact.get(key) != value:
                    errors.append(
                        f"manifest artifact.{key} must be {value!r}, got {artifact.get(key)!r}"
                    )
        lineage = manifest.get("lineage")
        if not isinstance(lineage, dict):
            errors.append("manifest lineage field must be an object")
        else:
            expected_lineage = {
                "relationship": "derived_from",
                "name": EXPECTED_NAME,
                "version": "1.0.0",
                "repository": UPSTREAM_REPOSITORY,
                "tag": "v1.0.0",
                "commit": UPSTREAM_COMMIT,
                "release": UPSTREAM_RELEASE,
                "archive_doi": UPSTREAM_DOI,
                "archive": UPSTREAM_ARCHIVE,
                "archive_landing_metadata_license": UPSTREAM_LANDING_METADATA_LICENSE,
                "archived_source_license": UPSTREAM_SOURCE_LICENSE,
                "release_date": "2026-06-29",
            }
            for key, value in expected_lineage.items():
                if lineage.get(key) != value:
                    errors.append(
                        f"manifest lineage.{key} must be {value!r}, got {lineage.get(key)!r}"
                    )
        required_paths_checked = _check_required_paths(
            root, manifest.get("required_paths"), errors
        )
    if not isinstance(schema, dict) or schema.get("$schema") != (
        "https://json-schema.org/draft/2020-12/schema"
    ):
        errors.append("artifact schema must declare JSON Schema draft 2020-12")

    pyproject_path = root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = pyproject.get("project", {})
        if project.get("name") != EXPECTED_PACKAGE:
            errors.append(f"pyproject project.name must be {EXPECTED_PACKAGE!r}")
        if project.get("version") != EXPECTED_VERSION:
            errors.append(f"pyproject project.version must be {EXPECTED_VERSION!r}")
        if _project_license(project) != EXPECTED_LICENSE:
            errors.append(f"pyproject license must be {EXPECTED_LICENSE!r}")
        project_urls = project.get("urls", {})
        if not isinstance(project_urls, dict):
            errors.append("pyproject project.urls must be a table")
        else:
            if project_urls.get("Repository") != EXPECTED_REPOSITORY:
                errors.append(
                    f"pyproject Repository URL must be {EXPECTED_REPOSITORY!r}"
                )
            if project_urls.get("Release") != EXPECTED_RELEASE:
                errors.append(f"pyproject Release URL must be {EXPECTED_RELEASE!r}")
            if project_urls.get("Upstream") != UPSTREAM_REPOSITORY:
                errors.append(f"pyproject Upstream URL must be {UPSTREAM_REPOSITORY!r}")
            if project_urls.get("Upstream-Release") != UPSTREAM_RELEASE:
                errors.append(
                    f"pyproject Upstream-Release URL must be {UPSTREAM_RELEASE!r}"
                )
            if project_urls.get("Upstream-Archive") != UPSTREAM_ARCHIVE:
                errors.append(
                    f"pyproject Upstream-Archive URL must be {UPSTREAM_ARCHIVE!r}"
                )
            if "Archive" in project_urls:
                errors.append(
                    "pyproject must not claim a separate v1.1.0 archive or DOI"
                )
        setuptools_config = pyproject.get("tool", {}).get("setuptools", {})
        license_files = setuptools_config.get("license-files", [])
        if not isinstance(license_files, list) or not {"LICENSE", "NOTICE"}.issubset(
            set(license_files)
        ):
            errors.append("pyproject must package both LICENSE and NOTICE")
    except FileNotFoundError:
        errors.append(f"missing file: {pyproject_path}")
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"invalid TOML in {pyproject_path}: {exc}")

    cff = _cff_top_level_scalars(root / "CITATION.cff", errors)
    for key, value in {
        "version": EXPECTED_VERSION,
        "date-released": EXPECTED_RELEASE_DATE,
        "license": EXPECTED_LICENSE,
        "repository-code": EXPECTED_REPOSITORY,
        "url": EXPECTED_RELEASE,
    }.items():
        if cff.get(key) != value:
            errors.append(f"CITATION.cff {key} must be {value!r}")
    if "doi" in cff:
        errors.append("CITATION.cff must not claim a top-level DOI for v1.1.0")

    license_path = root / "LICENSE"
    try:
        license_sha256 = hashlib.sha256(license_path.read_bytes()).hexdigest()
        if license_sha256 != OFFICIAL_LICENSE_SHA256:
            errors.append(
                "LICENSE must exactly match the official Apache License 2.0 plaintext"
            )
    except FileNotFoundError:
        errors.append(f"missing file: {license_path}")

    notice_path = root / "NOTICE"
    try:
        notice = notice_path.read_text(encoding="utf-8")
        for required_text in (UPSTREAM_REPOSITORY, UPSTREAM_COMMIT, "does not\nmodify the License"):
            if required_text not in notice:
                errors.append(f"NOTICE is missing required attribution: {required_text!r}")
    except FileNotFoundError:
        errors.append(f"missing file: {notice_path}")

    usage_policy_path = root / "USAGE_POLICY.md"
    try:
        usage_policy = usage_policy_path.read_text(encoding="utf-8")
        if "non-binding responsible-use guidance" not in usage_policy or (
            "does not\nrestrict or modify" not in usage_policy
        ):
            errors.append("USAGE_POLICY.md must state that its guidance does not modify the license")
    except FileNotFoundError:
        errors.append(f"missing file: {usage_policy_path}")

    package_init_path = root / "src" / "crosstaint" / "__init__.py"
    try:
        package_init = package_init_path.read_text(encoding="utf-8")
        if f'__version__ = "{EXPECTED_VERSION}"' not in package_init:
            errors.append(f"package __version__ must be {EXPECTED_VERSION!r}")
    except FileNotFoundError:
        errors.append(f"missing file: {package_init_path}")

    _check_smoke_benchmark(root / "examples" / "smoke-benchmark.json", errors)

    return {
        "ok": not errors,
        "artifact": EXPECTED_NAME,
        "version": EXPECTED_VERSION,
        "root": str(root),
        "required_paths_checked": required_paths_checked,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the repository-level artifact validator."""
    parser = argparse.ArgumentParser(
        description="Validate the CrossTaint v1.1.0 release structure and metadata"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path to a CrossTaint checkout (default: discover from current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the validation report as JSON",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Also run deterministic fixed-weight propagation over the bundled "
            "synthetic-oracle fixture"
        ),
    )
    args = parser.parse_args(argv)

    try:
        root = args.root.resolve() if args.root else _find_repository_root(Path.cwd())
        report = validate_release(root)
        if args.smoke and report["ok"]:
            try:
                report["smoke"] = run_synthetic_oracle_smoke(root)
            except Exception as exc:
                report["ok"] = False
                report["errors"].append(f"synthetic-oracle smoke failed: {exc}")
    except FileNotFoundError as exc:
        report = {
            "ok": False,
            "artifact": EXPECTED_NAME,
            "version": EXPECTED_VERSION,
            "errors": [str(exc)],
        }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["ok"]:
        print(
            f"CrossTaint {report['version']} artifact validation passed "
            f"({report['required_paths_checked']} required paths)."
        )
        if args.smoke:
            print(
                "Synthetic-oracle smoke passed "
                f"({report['smoke']['hops_recovered']} hops recovered)."
            )
    else:
        print("CrossTaint artifact validation failed:", file=sys.stderr)
        for error in report["errors"]:
            print(f"- {error}", file=sys.stderr)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
