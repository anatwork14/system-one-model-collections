"""Regression tests for M16 controlled-data provenance and output hashes."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREPARE = ROOT / "retrain/16_adas1_multiexit_4b/scripts/prepare_data.py"


class M16PrepareDataTest(unittest.TestCase):
    def test_manifest_pins_canonical_controlled_outputs_separately_from_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "controlled"
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--output", str(output)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"], {"train": 2268, "dev": 408})
            self.assertFalse(manifest["outer_324_eval_read"])
            self.assertEqual(manifest["controlled_sha256"], {
                "train": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
                "dev": hashlib.sha256((output / "dev.jsonl").read_bytes()).hexdigest(),
            })
            self.assertNotEqual(manifest["source_sha256"]["train"], manifest["controlled_sha256"]["train"])


if __name__ == "__main__":
    unittest.main()
