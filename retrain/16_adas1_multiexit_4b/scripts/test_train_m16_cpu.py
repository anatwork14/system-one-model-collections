"""CPU integration tests for the M16 training stage loops with a deterministic toy backbone."""
import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = ROOT / "retrain/16_adas1_multiexit_4b/scripts/train_m16.py"
METHOD_SRC = ROOT / "retrain/16_adas1_multiexit_4b/src"
sys.path.insert(0, str(METHOD_SRC))
spec = importlib.util.spec_from_file_location("train_m16", TRAINER_PATH)
train_m16 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_m16
spec.loader.exec_module(train_m16)


class M16TrainingLoopCpuTest(unittest.TestCase):
    def test_teacher_and_multiexit_stages_train_validate_and_save_weights(self):
        import torch
        from torch import nn
        from adas1.multiexit import ExitHeadBank, supervised_decision_loss
        from adas1.training import aggregate_multiexit_loss, summarize_decisions, target_index

        class TinyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(4, 2)

            def forward(self, hidden, input_ids, attention_mask, records, embedding):
                pooled = hidden[:, 0, :]
                return [[self.projection(pooled[index]) for index in range(len(records))]]

        class TinyBackbone:
            def __call__(self, input_ids, attention_mask, use_cache, output_hidden_states, return_dict):
                scalar = input_ids[:, :1].to(dtype=torch.float32).unsqueeze(-1)
                base = torch.cat([scalar + offset for offset in range(4)], dim=-1)
                layer1, layer2, layer3 = base, base * 1.1, base * 1.2
                states = (base * 0, layer1, layer2, layer3) if output_hidden_states else None
                return SimpleNamespace(last_hidden_state=layer3, hidden_states=states)

        class Tokenizer:
            pad_token_id = 0

        def encode(_tokenizer, row, max_length):
            encoded_question = SimpleNamespace(option_ids=("yes", "no"))
            return SimpleNamespace(input_ids=(int(row["token"]),), questions=(encoded_question,),
                                   record_id=row["id"])

        def collate(records, pad_id, device):
            ids = torch.tensor([[records[0].input_ids[0]]], dtype=torch.long, device=device)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "records": records}

        def row(record_id, family, token, target):
            return {"id": record_id, "family": family, "token": token,
                    "questions": {"decision": {"type": "choice", "target": target}}}

        runtime = {
            "torch": torch, "device": torch.device("cpu"), "dtype": torch.float32,
            "text_model": TinyBackbone(), "embedding_weight": torch.zeros((2, 4)),
            "tokenizer": Tokenizer(), "collate_records": collate, "encode_record": encode,
            "target_index": target_index, "supervised_loss": supervised_decision_loss,
            "aggregate_loss": aggregate_multiexit_loss, "summarize": summarize_decisions,
            "ExitHeadBank": ExitHeadBank, "head_factory": TinyHead,
            "head_config": {"hidden_size": 4}, "exit_layers": (1, 2),
            "train_rows": [row("t1", "train-a", 1, "yes"), row("t2", "train-b", 2, "no")],
            "dev_rows": [row("d1", "dev-a", 3, "yes")],
        }
        args = SimpleNamespace(
            _started_at=time.monotonic(), max_runtime_seconds=120, max_length=128,
            learning_rate=0.01, weight_decay=0.0, max_epochs=1, patience=1,
            gradient_accumulation=1, brier_weight=0.1, label_smoothing=0.0,
            distill_weight=0.5, temperature=2.0, ordinal_rps_weight=0.1,
            base_model="toy", revision="toy", seed=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            teacher_dir = root / "teacher"
            teacher_state = train_m16._train_stage(runtime, args, "teacher", teacher_dir)
            self.assertTrue((teacher_dir / "teacher_head.pt").is_file())
            self.assertTrue((teacher_dir / "dev_metrics.json").is_file())
            exit_dir = root / "multiexit"
            exit_state = train_m16._train_stage(runtime, args, "multiexit", exit_dir, teacher_state)
            self.assertTrue((exit_dir / "exit_heads.pt").is_file())
            self.assertTrue((exit_dir / "teacher_head.pt").is_file() is False)
            self.assertEqual(set(exit_state), {f"heads.{layer}.{key}"
                             for layer in (1, 2) for key in teacher_state})
            self.assertTrue((exit_dir / "dev_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
