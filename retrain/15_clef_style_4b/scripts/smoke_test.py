#!/usr/bin/env python3
"""Exercise the official Clef head with Qwen3.5-4B dimensions and mixed schemas."""

import json
import sys
from pathlib import Path

import torch


WORKSPACE = Path(__file__).resolve().parents[3]
REFERENCE = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
BASE_CONFIG = WORKSPACE / "shared/models/qwen3.5-4b/config.json"
sys.path.insert(0, str(REFERENCE))
from joint_schema_model import EncodedQuestion, EncodedRecord, JointSchemaHead  # noqa: E402


def main() -> None:
    config = json.loads(BASE_CONFIG.read_text())["text_config"]
    qwen_hidden = config["hidden_size"]
    head_config = json.loads((REFERENCE / "joint_head_config.json").read_text())
    head_config["hidden_size"] = qwen_hidden
    head = JointSchemaHead(**head_config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    record = EncodedRecord(
        input_ids=tuple(ids[0].tolist()),
        record_id="shape-smoke",
        questions=(
            EncodedQuestion("department", 0, (1, 3), ((3, 4), (4, 5)), ("billing", "technical")),
            EncodedQuestion("urgent", 2, (5, 7), ((7, 8), (8, 9)), ("true", "false")),
            EncodedQuestion("severity", 1, (9, 10), ((10, 11), (11, 12), (12, 13), (13, 14)), ("0", "1", "2", "3")),
        ),
    )
    hidden = torch.randn(1, ids.shape[1], qwen_hidden)
    mask = torch.ones_like(ids)
    # Synthetic token embeddings keep this a head shape check; no backbone weights are loaded.
    embedding = torch.randn(32, qwen_hidden)
    with torch.inference_mode():
        logits = head(hidden, ids, mask, [record], embedding)[0]
    shapes = [tuple(value.shape) for value in logits]
    expected = [(2,), (2,), (4,)]
    assert shapes == expected, f"expected {expected}, received {shapes}"
    assert all(torch.isfinite(value).all() for value in logits)
    print(f"Qwen3.5-4B hidden_size={qwen_hidden}; mixed-question logits={shapes}; finite=True")


if __name__ == "__main__":
    main()
