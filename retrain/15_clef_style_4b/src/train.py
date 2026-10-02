#!/usr/bin/env python3
"""Train a fresh 4B Clef-style head, optionally with explicitly chosen LoRA."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model


HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
WORKSPACE = METHOD.parents[1]
REFERENCE = WORKSPACE / "released/05_clef_flash_9b/weights/architecture"
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(HERE))
from clef4b.losses import option_target, question_loss  # noqa: E402
from joint_schema_model import (  # noqa: E402
    ClefModel,
    JointSchemaHead,
    collate_records,
    encode_record,
)


class RecordDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = encode_record(self.tokenizer, row, max_length=self.max_length)
        return row, encoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=METHOD / "data/train/nimble.jsonl")
    parser.add_argument("--base-model", type=Path, default=WORKSPACE / "shared/models/qwen3.5-4b")
    parser.add_argument("--output", type=Path, default=METHOD / "checkpoints/head_only")
    parser.add_argument("--stage", choices=("head_only", "lora"), default="head_only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", action="store_true", help="Explicitly allow extremely slow CPU training")
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap; 0 means all data")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.cpu:
        raise SystemExit("CUDA is unavailable. Use a supported CUDA host or pass --cpu for a slow feasibility run.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(args.base_model)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    backbone.config.use_cache = False
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if args.stage == "lora":
        # This explicit broad target list is a local design choice. Cloudflare did not
        # publish the exact adapter target list. It covers Qwen3.5 attention, MLP,
        # and linear-attention projections while leaving the vision tower frozen.
        lora = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
            ],
            modules_to_save=None,
        )
        backbone = get_peft_model(backbone, lora)
        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head_config = json.loads((REFERENCE / "joint_head_config.json").read_text())
    head_config["hidden_size"] = backbone.config.text_config.hidden_size
    head = JointSchemaHead(**head_config)
    model = ClefModel(backbone, head).to(device=device, dtype=dtype)
    dataset = RecordDataset(args.data, processor.tokenizer, args.max_length)
    if args.max_records:
        dataset.rows = dataset.rows[: args.max_records]
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda rows: rows)
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be at least 1")
    head_params = list(model.head.parameters())
    lora_params = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head}]
        + ([{"params": lora_params, "lr": args.lr_lora}] if lora_params else []),
        weight_decay=0.01,
    )
    model.train()
    if args.stage == "head_only":
        model.language_model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        total = 0.0
        correct = count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, rows in enumerate(loader):
            row, encoded = rows[0]
            batch = collate_records([encoded], processor.tokenizer.pad_token_id, device)
            predictions = model(batch)[0]
            terms = []
            for (question_id, question), logits, encoded_question in zip(
                row["questions"].items(), predictions, encoded.questions
            ):
                target_index = option_target(question, encoded_question.option_ids)
                combined, _, _ = question_loss(
                    logits,
                    target_index,
                    label_smoothing=args.label_smoothing,
                    brier_weight=args.brier_weight,
                )
                terms.append(combined)
                correct += int(logits.detach().argmax().item() == target_index)
                count += 1
            loss = torch.stack(terms).mean()
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
        print(json.dumps({"epoch": epoch + 1, "mean_record_loss": total / max(1, len(loader)), "question_accuracy": correct / max(1, count), "records": len(dataset)}))
        torch.save(model.head.state_dict(), args.output / "joint_head.pt")
        if args.stage == "lora":
            model.language_model.save_pretrained(args.output / "adapter")
    metadata = vars(args).copy()
    metadata["base_model"] = str(args.base_model)
    metadata["status"] = "local_reproduction"
    metadata["lora_target_modules"] = "Qwen3.5 language projections listed in train.py; local choice, not Cloudflare-disclosed"
    (args.output / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
