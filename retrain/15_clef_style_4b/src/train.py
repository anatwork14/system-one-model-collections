#!/usr/bin/env python3
"""Train a fresh 4B Clef-style head, optionally with explicitly chosen LoRA."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model


HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
WORKSPACE = METHOD.parents[1]
REFERENCE = METHOD / "upstream/clef_reference" if (METHOD / "upstream/clef_reference").exists() else (WORKSPACE / "released/05_clef_flash_9b/weights/architecture")
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
    parser.add_argument("--augmented", type=Path,
                        help="Optional train-only schema variants generated from the canonical train split")
    parser.add_argument("--validation", type=Path, default=METHOD / "data/dev/nimble.jsonl")
    parser.add_argument("--base-model", default=str(WORKSPACE / "shared/models/qwen3.5-4b") if (WORKSPACE / "shared/models/qwen3.5-4b").exists() else "Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--output", type=Path, default=METHOD / "checkpoints/head_only_clean")
    parser.add_argument("--stage", choices=("head_only", "lora"), default="head_only")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=2e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu", action="store_true", help="Explicitly allow extremely slow CPU training")
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap; 0 means all data")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.cpu:
        raise SystemExit("CUDA is unavailable. Use a supported CUDA host or pass --cpu for a slow feasibility run.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    train_dataset = RecordDataset(args.data, AutoTokenizer.from_pretrained(
        args.base_model, revision=args.revision), args.max_length)
    base_train_count = len(train_dataset.rows)
    if args.augmented:
        variant_rows = [json.loads(line) for line in args.augmented.read_text().splitlines() if line.strip()]
        train_dataset.rows.extend(variant_rows)
    validation_dataset = RecordDataset(args.validation, train_dataset.tokenizer, args.max_length)
    train_ids = {row["id"] for row in train_dataset.rows}
    dev_ids = {row["id"] for row in validation_dataset.rows}
    train_families = {row["family"] for row in train_dataset.rows}
    dev_families = {row["family"] for row in validation_dataset.rows}
    if not train_dataset.rows or not validation_dataset.rows or train_ids & dev_ids or train_families & dev_families:
        raise ValueError("Training and inner-dev data must be nonempty and disjoint by id and family")
    if args.epochs < 1 or args.patience < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs, patience, and gradient accumulation must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = train_dataset.tokenizer
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model,
        revision=args.revision,
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
    dataset = train_dataset
    if args.max_records:
        dataset.rows = dataset.rows[: args.max_records]
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda rows: rows)
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be at least 1")
    head_params = list(model.head.parameters())
    lora_params = [parameter for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.lr_head}]
        + ([{"params": lora_params, "lr": args.lr_lora}] if lora_params else []),
        weight_decay=0.01,
    )
    model.train()
    if args.stage == "head_only":
        model.language_model.eval()
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    def capture_trainable_state():
        return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                if parameter.requires_grad}

    def evaluate_dev():
        model.eval()
        loss_sum = 0.0
        correct = question_count = 0
        with torch.inference_mode():
            for rows in validation_loader:
                row, encoded = rows[0]
                batch = collate_records([encoded], tokenizer.pad_token_id, device)
                predictions = model(batch)[0]
                terms = []
                for question, logits, encoded_question in zip(row["questions"].values(), predictions, encoded.questions):
                    target_index = option_target(question, encoded_question.option_ids)
                    loss, _, _ = question_loss(logits, target_index,
                        label_smoothing=args.label_smoothing, brier_weight=args.brier_weight)
                    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                        raise FloatingPointError("Non-finite Clef-style dev score/loss")
                    terms.append(loss)
                    correct += int(logits.argmax().item() == target_index)
                    question_count += 1
                loss_sum += float(torch.stack(terms).mean())
        return {"mean_record_loss": loss_sum / max(len(validation_dataset), 1),
                "question_accuracy": correct / max(question_count, 1),
                "records": len(validation_dataset), "questions": question_count}

    for epoch in range(args.epochs):
        model.train()
        if args.stage == "head_only":
            model.language_model.eval()
        total = 0.0
        correct = count = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, rows in enumerate(loader):
            row, encoded = rows[0]
            batch = collate_records([encoded], tokenizer.pad_token_id, device)
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
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Clef-style training loss at record {batch_index}")
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            if (batch_index + 1) % 50 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} batch {batch_index + 1}/{len(loader)} mean_loss {total / (batch_index + 1):.4f} acc {correct / max(1, count):.4f}", flush=True)
        dev_metrics = evaluate_dev()
        row = {"epoch": epoch + 1, "train_mean_record_loss": total / max(1, len(loader)),
               "train_question_accuracy": correct / max(1, count), "train_records": len(dataset), "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_metrics["mean_record_loss"] < best_loss:
            best_loss = dev_metrics["mean_record_loss"]
            best_epoch = epoch + 1
            best_state = capture_trainable_state()
            stale_epochs = 0
            torch.save(best_state, args.output / "best_trainable_state.pt")
        else:
            stale_epochs += 1
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No finite Clef-style inner-dev checkpoint was selected")
    named_parameters = dict(model.named_parameters())
    for name, value in best_state.items():
        named_parameters[name].data.copy_(value.to(device=device, dtype=named_parameters[name].dtype))
    torch.save(model.head.state_dict(), args.output / "joint_head.pt")
    if args.stage == "lora":
        model.language_model.save_pretrained(args.output / "adapter")
    (args.output / "dev_eval.json").write_text(json.dumps(evaluate_dev(), indent=2) + "\n", encoding="utf-8")
    metadata = vars(args).copy()
    metadata["base_model"] = str(args.base_model)
    metadata["status"] = "local_reproduction"
    metadata["best_epoch"] = best_epoch
    metadata["best_inner_dev_loss"] = best_loss
    metadata["selection_metric"] = "inner_dev_mean_record_loss"
    metadata["train_records"] = len(dataset)
    metadata["base_train_records"] = base_train_count
    metadata["augmented_train_records"] = len(dataset) - base_train_count
    metadata["dev_records"] = len(validation_dataset)
    metadata["train_families"] = sorted(train_families)
    metadata["dev_families"] = sorted(dev_families)
    metadata["outer_eval_used_for_training_or_selection"] = False
    metadata["lora_target_modules"] = "Qwen3.5 language projections listed in train.py; local choice, not Cloudflare-disclosed"
    (args.output / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
