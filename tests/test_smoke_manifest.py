from __future__ import annotations

import unittest
from pathlib import Path

from crosstaint.experiments.corpus import load_benchmark_dataset


ROOT = Path(__file__).resolve().parents[1]


class SmokeManifestTests(unittest.TestCase):
    def test_synthetic_smoke_manifest_loads_without_external_data(self) -> None:
        dataset = load_benchmark_dataset(
            ROOT / "examples" / "smoke-benchmark.json",
            require_two_source_ground_truth=False,
        )
        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(len(dataset.edges), 2)
        self.assertEqual(len(dataset.nodes), 3)
        self.assertEqual(dataset.cases[0].source_count, 0)
        self.assertEqual(dataset.metadata["evidence_scope"], "synthetic_oracle")
        self.assertFalse(dataset.metadata["oracle_metadata"]["independent_provenance"])
        self.assertEqual(
            dataset.cases[0].ground_truth_addresses,
            [
                "0x2222222222222222222222222222222222222222",
                "0x3333333333333333333333333333333333333333",
            ],
        )


if __name__ == "__main__":
    unittest.main()
