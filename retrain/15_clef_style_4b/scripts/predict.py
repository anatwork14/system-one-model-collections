#!/usr/bin/env python3
"""Run checkpoint inference and emit normalized per-question decision records."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
sys.path.insert(0, str(METHOD / "src"))
sys.path.insert(0, str(HERE))
from train import RecordDataset, WORKSPACE, option_target  # noqa: E402
from clef4b.model import ClefModel  # noqa: E402
from clef4b.collator import collate_records  # noqa: E402
from clef4b.joint_head import JointSchemaHead  # noqa: E402

REFERENCE = METHOD / "upstream/clef_reference" if (METHOD / "upstream/clef_reference").exists() else (WORKSPACE / "released/05_clef_flash_9b/weights/architecture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=METHOD / "data/dev/nimble.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "run_config.json").read_text(encoding="utf-8"))
    base_model = str(config["base_model"])
    revision = str(config["revision"])
    device = torch.device(args.device)
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    dataset = RecordDataset(args.input, tokenizer, int(config["max_length"]))
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    adapter = args.checkpoint / "adapter"
    if config["stage"] == "lora":
        if not adapter.is_dir():
            raise FileNotFoundError(f"LoRA checkpoint adapter is missing: {adapter}")
        backbone = PeftModel.from_pretrained(backbone, adapter, is_trainable=False)
    elif config["stage"] != "head_only":
        raise ValueError(f"unsupported saved CLEF stage: {config['stage']}")
    head_config = json.loads((REFERENCE / "joint_head_config.json").read_text(encoding="utf-8"))
    head_config["hidden_size"] = backbone.config.text_config.hidden_size
    head = JointSchemaHead(**head_config)
    head_state = torch.load(args.checkpoint / "joint_head.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(head_state, strict=True)
    model = ClefModel(backbone, head).to(device=device, dtype=dtype).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for index, row in enumerate(dataset.rows):
            start = time.perf_counter()
            encoded = dataset[index][1]
            batch = collate_records([encoded], tokenizer.pad_token_id, device)
            question_logits = model(batch)[0]
            latency_ms = (time.perf_counter() - start) * 1000.0
            if len(question_logits) != len(encoded.questions):
                raise ValueError(f"question/logit count differs for record {row['id']}")
            for (question_id, question), logits, encoded_question in zip(
                    row["questions"].items(), question_logits, encoded.questions):
                logits = logits.float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"non-finite logits for {row['id']}/{question_id}")
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                options = list(encoded_question.option_ids)
                if len(options) != len(probabilities) or len(options) < 2:
                    raise ValueError(f"invalid option/logit count for {row['id']}/{question_id}")
                probability_map = {option: float(value) for option, value in zip(options, probabilities)}
                target_index = option_target(question, options)
                prediction = max(probability_map, key=probability_map.get)
                result = {
                    "id": f"{row['id']}/{question_id}", "request_id": str(row["id"]),
                    "method": f"M15-{config['stage']}",
                    "base_model": base_model, "checkpoint_revision": revision,
                    "dataset": "nimble", "split": "inner_dev", "label": options[target_index],
                    "input_tokens": len(encoded.input_ids), "num_options": len(options),
                    "predicted_option": prediction, "probabilities": probability_map,
                    "latency_ms": latency_ms, "inference_success": True,
                }
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
