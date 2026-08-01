from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # CI installs the schema-conformance test dependency.
    Draft202012Validator = None
    FormatChecker = None

from crosstaint.artifact import run_synthetic_oracle_smoke, validate_release


ROOT = Path(__file__).resolve().parents[1]


class ArtifactValidationTests(unittest.TestCase):
    def test_release_manifest_and_required_paths(self) -> None:
        report = validate_release(ROOT)
        self.assertTrue(report["ok"], report["errors"])
        self.assertGreaterEqual(report["required_paths_checked"], 20)

    @unittest.skipUnless(Draft202012Validator, "jsonschema is not installed")
    def test_release_manifest_validates_against_declared_schema(self) -> None:
        assert Draft202012Validator is not None
        assert FormatChecker is not None
        schema = json.loads(
            (ROOT / "artifact" / "manifest.schema.json").read_text(encoding="utf-8")
        )
        instance = json.loads(
            (ROOT / "artifact" / "manifest.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)

    @unittest.skipUnless(Draft202012Validator, "jsonschema is not installed")
    def test_smoke_manifest_validates_against_benchmark_schema(self) -> None:
        assert Draft202012Validator is not None
        assert FormatChecker is not None
        schema = json.loads(
            (ROOT / "schemas" / "benchmark-manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        instance = json.loads(
            (ROOT / "examples" / "smoke-benchmark.json").read_text(encoding="utf-8")
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(instance)

    def test_installed_cli_smoke_implementation(self) -> None:
        report = run_synthetic_oracle_smoke(ROOT)
        self.assertTrue(report["ok"])
        self.assertEqual(report["evidence_scope"], "synthetic_oracle")
        self.assertFalse(report["independent_provenance"])
        self.assertEqual(report["hops_recovered"], 2)


if __name__ == "__main__":
    unittest.main()
