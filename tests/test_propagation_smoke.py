from __future__ import annotations

import unittest
from pathlib import Path

from crosstaint.config import Config
from crosstaint.experiments.corpus import load_benchmark_dataset
from crosstaint.propagation.engine import PropagationEngine


ROOT = Path(__file__).resolve().parents[1]


class PropagationSmokeTests(unittest.TestCase):
    def tearDown(self) -> None:
        Config.reset()

    def test_fixed_weight_propagation_recovers_synthetic_two_hop_path(self) -> None:
        Config.reset()
        config = Config.load(ROOT / "config")
        dataset = load_benchmark_dataset(
            ROOT / "examples" / "smoke-benchmark.json",
            require_two_source_ground_truth=False,
        )
        case = dataset.cases[0]
        engine = PropagationEngine.from_config(config)

        result = engine.propagate(
            origin=case.origin_node_id,
            origin_chain=case.origin_chain,
            origin_value=case.origin_value,
            graph=dataset.nodes,
            edges=dataset.edges,
            case_id=case.case_id,
        )

        self.assertEqual(
            [entry.address for entry in result.suspect_set],
            case.ground_truth_addresses,
        )
        self.assertEqual([entry.hop_distance for entry in result.suspect_set], [1, 2])
        self.assertEqual(result.config_hash, Config.config_hash())
        self.assertFalse(result.soundness_certificate.parameters_loaded)


if __name__ == "__main__":
    unittest.main()
