from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crosstaint.config import Config


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DEFAULTS = ROOT / "src" / "crosstaint" / "default_config"


class ConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        Config.reset()

    def test_source_and_packaged_defaults_are_equivalent(self) -> None:
        source = Config.load(ROOT / "config")
        source_hash = Config.config_hash()
        source_counts = (len(source.chains), len(source.bridges))

        Config.reset()
        packaged = Config.load(PACKAGE_DEFAULTS)
        self.assertEqual(Config.config_hash(), source_hash)
        self.assertEqual((len(packaged.chains), len(packaged.bridges)), source_counts)
        self.assertEqual(source_counts, (6, 9))

    def test_explicit_missing_config_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with self.assertRaises(FileNotFoundError):
                Config.load(missing)


if __name__ == "__main__":
    unittest.main()
