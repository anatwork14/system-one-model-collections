"""CPU-testable objectives for M16 intermediate-exit training."""

import hashlib
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD / "src"))


class MultiExitLossTest(unittest.TestCase):
    def api(self):
        spec = importlib.util.find_spec("adas1.multiexit")
        self.assertIsNotNone(spec, "M16 multi-exit loss module has not been implemented")
        return importlib.import_module("adas1.multiexit")

    def test_distillation_is_zero_for_identical_teacher_and_student(self):
        import torch
        loss = self.api().distillation_kl_loss
        logits = torch.tensor([0.2, -0.3, 1.1], dtype=torch.float64)
        self.assertAlmostEqual(float(loss(logits, logits, temperature=2.0)), 0.0, places=12)

    def test_distillation_backpropagates_to_student_only(self):
        import torch
        loss = self.api().distillation_kl_loss
        student = torch.tensor([0.3, -0.2, 0.8], requires_grad=True)
        teacher = torch.tensor([0.1, 0.4, -0.5], requires_grad=True)
        value = loss(student, teacher, temperature=2.0)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_multiexit_loss_promotes_bfloat16_logits_to_float32(self):
        import torch
        objective = self.api().multiexit_question_loss
        student = torch.tensor([0.8, -0.4], dtype=torch.bfloat16, requires_grad=True)
        teacher = torch.tensor([0.2, 0.1], dtype=torch.bfloat16, requires_grad=True)
        value = objective(student, teacher, target=0, question_type="choice")
        self.assertEqual(value.dtype, torch.float32)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)

    def test_ordinal_rps_penalizes_reversed_score_order(self):
        import torch
        rps = self.api().ordinal_rps_loss
        preferred = torch.tensor([5.0, 0.0, -5.0])
        reversed_order = torch.tensor([-5.0, 0.0, 5.0])
        self.assertLess(float(rps(preferred, 0)), float(rps(reversed_order, 0)))

    def test_score_loss_adds_ordinal_term_but_choice_loss_does_not(self):
        import torch
        objective = self.api().multiexit_question_loss
        student = torch.tensor([0.2, -0.1, 0.4])
        teacher = torch.tensor([-0.2, 0.3, 0.1])
        choice = objective(student, teacher, target=1, question_type="choice", ordinal_rps_weight=0.4)
        score = objective(student, teacher, target=1, question_type="score", ordinal_rps_weight=0.4)
        self.assertGreater(float(score), float(choice))

    def test_supervised_loss_supports_m15_label_smoothing(self):
        import torch
        import torch.nn.functional as F
        loss = self.api().supervised_decision_loss
        logits = torch.tensor([0.3, -0.4, 0.8])
        target = 1
        smoothing = 0.2
        probabilities = logits.softmax(dim=-1)
        one_hot = F.one_hot(torch.tensor(target), num_classes=3).to(logits.dtype)
        expected = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]), label_smoothing=smoothing)
        expected = expected + 0.1 * (probabilities - one_hot).square().sum()
        self.assertAlmostEqual(float(loss(logits, target, label_smoothing=smoothing)), float(expected), places=6)

    def test_exit_head_bank_routes_the_requested_hidden_state_per_layer(self):
        import torch
        from torch import nn
        module = self.api()
        self.assertTrue(hasattr(module, "ExitHeadBank"), "exit-head bank has not been implemented")

        class SumHead(nn.Module):
            def forward(self, hidden, input_ids, mask, records, embedding):
                return hidden.sum()

        bank = module.ExitHeadBank((12, 16), SumHead)
        outputs = bank(
            hidden_states={12: torch.tensor([1.0]), 16: torch.tensor([2.0])},
            input_ids=torch.zeros((1, 1), dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            records=[],
            output_embedding_weight=torch.zeros((1, 1)),
        )
        self.assertEqual(set(outputs), {12, 16})
        self.assertEqual(float(outputs[12]), 1.0)
        self.assertEqual(float(outputs[16]), 2.0)

    def test_target_index_preserves_typed_option_mapping(self):
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import target_index
        self.assertEqual(target_index({"type": "noul", "target": False}, ("true", "false")), 1)
        self.assertEqual(target_index({"type": "score", "target": 2}, ("0", "1", "2")), 2)

    def test_multiexit_loss_handles_variable_option_counts_and_stops_teacher_gradient(self):
        import torch
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import aggregate_multiexit_loss
        student_binary = torch.tensor([0.8, -0.4], requires_grad=True)
        student_score = torch.tensor([0.1, -0.2, 0.5], requires_grad=True)
        teacher_binary = torch.tensor([0.2, 0.1], requires_grad=True)
        teacher_score = torch.tensor([-0.2, 0.0, 0.8], requires_grad=True)
        total, per_exit = aggregate_multiexit_loss(
            {12: [[student_binary, student_score]], 32: [[teacher_binary, teacher_score]]},
            targets=[[0, 2]], question_types=[["noul", "score"]], teacher_layer=32,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(per_exit), {12})
        total.backward()
        self.assertIsNotNone(student_binary.grad)
        self.assertIsNotNone(student_score.grad)
        self.assertIsNone(teacher_binary.grad)
        self.assertIsNone(teacher_score.grad)

    def test_aggregate_loss_rejects_question_count_mismatch(self):
        import torch
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import aggregate_multiexit_loss
        with self.assertRaisesRegex(ValueError, "question counts"):
            aggregate_multiexit_loss(
                {12: [[torch.tensor([0.1, 0.2])]], 32: [[torch.tensor([0.1, 0.2])]]},
                targets=[[0, 1]], question_types=[["choice", "choice"]], teacher_layer=32,
            )

    def test_exit_layer_parser_requires_unique_increasing_intermediate_layers(self):
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import parse_exit_layers
        self.assertEqual(parse_exit_layers("12,16,20"), (12, 16, 20))
        for invalid in ("16,12", "12,12", "0,12", "12,32", "", "x,16"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_exit_layers(invalid)

    def test_decision_metrics_support_variable_option_counts(self):
        import math
        import torch
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import summarize_decisions
        result = summarize_decisions(
            [torch.log(torch.tensor([0.8, 0.2])), torch.log(torch.tensor([0.1, 0.8, 0.1]))],
            [0, 2], bins=1,
        )
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.5, places=6)
        self.assertAlmostEqual(result["mean_nll"], (-math.log(0.8) - math.log(0.1)) / 2, places=6)
        self.assertAlmostEqual(result["ece"], 0.3, places=6)

    def test_controlled_loader_never_reads_outer_test_and_checks_manifest_counts(self):
        import json
        import tempfile
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import load_controlled_splits
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            train = {"id": "tr1", "family": "train-a", "source_id": "src-tr1"}
            dev = {"id": "dv1", "family": "dev-a", "source_id": "src-dv1"}
            (data_dir / "train.jsonl").write_text(json.dumps(train) + "\n")
            (data_dir / "dev.jsonl").write_text(json.dumps(dev) + "\n")
            (data_dir / "outer_test.jsonl").write_text("not valid json; must remain unread\n")
            (data_dir / "manifest.json").write_text(json.dumps({
                "dataset": "nimble-m16-p0-controlled-v1",
                "counts": {"train": 1, "dev": 1},
                "outer_324_eval_read": False,
                "controlled_sha256": {
                    "train": hashlib.sha256((data_dir / "train.jsonl").read_bytes()).hexdigest(),
                    "dev": hashlib.sha256((data_dir / "dev.jsonl").read_bytes()).hexdigest(),
                },
            }))
            loaded_train, loaded_dev = load_controlled_splits(data_dir)
            self.assertEqual([row["id"] for row in loaded_train], ["tr1"])
            self.assertEqual([row["id"] for row in loaded_dev], ["dv1"])

    def test_controlled_loader_rejects_source_hash_mismatch(self):
        import json
        import tempfile
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import load_controlled_splits
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            train = {"id": "tr1", "family": "train-a", "source_id": "src-tr1"}
            dev = {"id": "dv1", "family": "dev-a", "source_id": "src-dv1"}
            (data_dir / "train.jsonl").write_text(json.dumps(train) + "\\n")
            (data_dir / "dev.jsonl").write_text(json.dumps(dev) + "\\n")
            (data_dir / "manifest.json").write_text(json.dumps({
                "dataset": "nimble-m16-p0-controlled-v1",
                "counts": {"train": 1, "dev": 1},
                "outer_324_eval_read": False,
                "controlled_sha256": {"train": "0" * 64, "dev": "0" * 64},
            }))
            (data_dir / "train.jsonl").write_text(json.dumps(train) + chr(10))
            (data_dir / "dev.jsonl").write_text(json.dumps(dev) + chr(10))
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_controlled_splits(data_dir)

    def test_controlled_loader_rejects_family_leakage(self):
        import json
        import tempfile
        spec = importlib.util.find_spec("adas1.training")
        self.assertIsNotNone(spec, "M16 training utilities have not been implemented")
        from adas1.training import load_controlled_splits
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            train = {"id": "tr1", "family": "shared", "source_id": "src-tr1"}
            dev = {"id": "dv1", "family": "shared", "source_id": "src-dv1"}
            (data_dir / "train.jsonl").write_text(json.dumps(train) + "\n")
            (data_dir / "dev.jsonl").write_text(json.dumps(dev) + "\n")
            (data_dir / "manifest.json").write_text(json.dumps({
                "dataset": "nimble-m16-p0-controlled-v1",
                "counts": {"train": 1, "dev": 1},
                "outer_324_eval_read": False,
                "controlled_sha256": {
                    "train": hashlib.sha256((data_dir / "train.jsonl").read_bytes()).hexdigest(),
                    "dev": hashlib.sha256((data_dir / "dev.jsonl").read_bytes()).hexdigest(),
                },
            }))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_controlled_splits(data_dir)

    def test_objective_rejects_incompatible_student_teacher_option_counts(self):
        import torch
        objective = self.api().multiexit_question_loss
        with self.assertRaisesRegex(ValueError, "same option count"):
            objective(torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2, 0.3]),
                      target=0, question_type="choice")


if __name__ == "__main__":
    unittest.main()
