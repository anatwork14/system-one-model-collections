"""CLI coverage for the runnable M16 stages."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_training_methods.py"


class M16TrainingCliTest(unittest.TestCase):
    def test_m16_teacher_stage_is_advertised_and_parseable(self):
        result = subprocess.run(
            [sys.executable, str(CLI), "M16", "--stage", "teacher", "--help"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_m16_multiexit_stage_is_advertised_and_parseable(self):
        result = subprocess.run(
            [sys.executable, str(CLI), "M16", "--stage", "multiexit", "--help"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_training_entrypoint_exposes_bounded_stages_without_loading_runtime(self):
        trainer = ROOT / "retrain/16_adas1_multiexit_4b/scripts/train_m16.py"
        result = subprocess.run(
            [sys.executable, str(trainer), "--help"], cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stage", result.stdout)
        self.assertIn("--max-epochs", result.stdout)
        self.assertIn("teacher", result.stdout)
        self.assertIn("multiexit", result.stdout)

    def test_multiexit_stage_rejects_missing_teacher_before_runtime_import(self):
        trainer = ROOT / "retrain/16_adas1_multiexit_4b/scripts/train_m16.py"
        result = subprocess.run(
            [sys.executable, str(trainer), "--stage", "multiexit"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--teacher-checkpoint is required", result.stderr)
        self.assertNotIn("torch", result.stderr)

    def test_multiexit_stage_rejects_missing_teacher_checkpoint_before_runtime_import(self):
        trainer = ROOT / "retrain/16_adas1_multiexit_4b/scripts/train_m16.py"
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "not-a-checkpoint.pt"
            result = subprocess.run(
                [sys.executable, str(trainer), "--stage", "multiexit", "--teacher-checkpoint", str(missing)],
                cwd=ROOT, text=True, capture_output=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("teacher checkpoint not found", result.stderr)
        self.assertNotIn("torch", result.stderr)

    def test_list_identifies_new_stages_and_remaining_deferred_stages(self):
        result = subprocess.run(
            [sys.executable, str(CLI), "--list"], cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("runnable-stage\tM16\tretrain/16_adas1_multiexit_4b\tteacher", result.stdout)
        self.assertIn("runnable-stage\tM16\tretrain/16_adas1_multiexit_4b\tmultiexit", result.stdout)
        self.assertIn("deferred\tM16-gate/serve", result.stdout)


if __name__ == "__main__":
    unittest.main()
