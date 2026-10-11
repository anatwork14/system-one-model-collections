#!/usr/bin/env python3
"""Train a full-depth M16 teacher and distillation-trained intermediate exits."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teacher", "multiexit", "full"), default="full")
    parser.add_argument("--data-dir", type=Path, default=METHOD / "data/controlled")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("M16_OUTPUT", METHOD / "checkpoints/m16")))
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--head-config", type=Path, default=Path(os.environ.get(
        "M16_HEAD_CONFIG", METHOD / "vendor/architecture/joint_head_config.json")))
    parser.add_argument("--exit-layers", default="12,16,20")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-runtime-seconds", type=int, default=14400,
                        help="hard wall-clock limit for all stages in this process")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ordinal-rps-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_epochs < 1 or args.patience < 1 or args.gradient_accumulation < 1:
        raise ValueError("max-epochs, patience, and gradient-accumulation must be positive")
    if args.max_length < 128 or args.max_runtime_seconds < 1:
        raise ValueError("max-length must be >=128 and max-runtime-seconds must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")
    if args.brier_weight < 0 or args.distill_weight < 0 or args.ordinal_rps_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if not 0 <= args.label_smoothing < 1 or args.temperature <= 0:
        raise ValueError("label-smoothing must be in [0,1) and temperature must be positive")
    if args.stage == "multiexit":
        if args.teacher_checkpoint is None:
            raise ValueError("--teacher-checkpoint is required for the multiexit stage")
        if not args.teacher_checkpoint.is_file():
            raise ValueError(f"teacher checkpoint not found: {args.teacher_checkpoint}")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(torch, value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_runtime(args):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    helper = Path(os.environ.get("M16_MODEL_HELPER", METHOD / "vendor/architecture/joint_schema_model.py"))
    if not helper.is_file():
        raise FileNotFoundError(f"M16 schema helper not found: {helper}")
    if not args.head_config.is_file():
        raise FileNotFoundError(f"M16 joint head config not found: {args.head_config}")
    sys.path.insert(0, str(METHOD / "src"))
    sys.path.insert(0, str(helper.parent))
    from adas1.multiexit import ExitHeadBank
    from adas1.training import (aggregate_multiexit_loss, load_controlled_splits,
                                parse_exit_layers, summarize_decisions, target_index)
    from adas1.multiexit import supervised_decision_loss
    from joint_schema_model import JointSchemaHead, collate_records, encode_record

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the 4B M16 trainer on CPU")
    torch.cuda.init()
    torch.empty((1,), device="cuda")
    torch.cuda.synchronize()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    train_rows, dev_rows = load_controlled_splits(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    backbone.config.use_cache = False
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    text_model = backbone.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    text_model.eval()
    head_config = json.loads(args.head_config.read_text(encoding="utf-8"))
    head_config["hidden_size"] = int(backbone.config.text_config.hidden_size)
    number_of_layers = int(backbone.config.text_config.num_hidden_layers)
    exit_layers = parse_exit_layers(args.exit_layers, number_of_layers)
    embedding_weight = backbone.get_output_embeddings().weight
    return {
        "torch": torch, "device": device, "dtype": dtype, "tokenizer": tokenizer,
        "backbone": backbone, "text_model": text_model, "embedding_weight": embedding_weight,
        "head_config": head_config, "head_factory": lambda: JointSchemaHead(**head_config),
        "exit_layers": exit_layers, "train_rows": train_rows, "dev_rows": dev_rows,
        "collate_records": collate_records, "encode_record": encode_record,
        "target_index": target_index, "supervised_loss": supervised_decision_loss,
        "aggregate_loss": aggregate_multiexit_loss, "ExitHeadBank": ExitHeadBank,
        "summarize": summarize_decisions,
    }


def _device_batch(runtime, row, max_length: int):
    if row.get("images") or row.get("videos"):
        raise ValueError("controlled M16 trainer expects text-only records; media rows are unsupported")
    encoded = runtime["encode_record"](runtime["tokenizer"], row, max_length=max_length)
    pad_id = runtime["tokenizer"].pad_token_id
    batch = runtime["collate_records"]([encoded], pad_id if pad_id is not None else 0, runtime["device"])
    targets = [runtime["target_index"](question, encoded_question.option_ids)
               for question, encoded_question in zip(row["questions"].values(), encoded.questions)]
    types = [str(question["type"]) for question in row["questions"].values()]
    if len(targets) != len(encoded.questions) or not targets:
        raise ValueError(f"record {row.get('id')} has a question/encoding mismatch")
    return encoded, batch, targets, types


def _base_hidden(runtime, batch, intermediate: bool):
    with runtime["torch"].no_grad():
        output = runtime["text_model"](
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            use_cache=False, output_hidden_states=intermediate, return_dict=True,
        )
    return output


def _teacher_logits(runtime, head, output, batch, encoded):
    return head(output.last_hidden_state, batch["input_ids"], batch["attention_mask"],
                [encoded], runtime["embedding_weight"])


def _multi_logits(runtime, bank, teacher_head, output, batch, encoded, exit_layers):
    hidden = {layer: output.hidden_states[layer] for layer in exit_layers}
    students = bank(hidden, batch["input_ids"], batch["attention_mask"], [encoded], runtime["embedding_weight"])
    teacher = _teacher_logits(runtime, teacher_head, output, batch, encoded)
    outputs = dict(students)
    outputs[32] = teacher
    return outputs


def _evaluate(runtime, args, rows, mode: str, head, teacher_head=None, bank=None, exit_layers=()):
    torch = runtime["torch"]
    layers = (32,) if mode == "teacher" else tuple(exit_layers) + (32,)
    all_logits = {layer: [] for layer in layers}
    all_targets = []
    for index, row in enumerate(rows, 1):
        if time.monotonic() - args._started_at > args.max_runtime_seconds:
            raise TimeoutError("M16 max-runtime-seconds reached during dev evaluation")
        encoded, batch, targets, _types = _device_batch(runtime, row, args.max_length)
        output = _base_hidden(runtime, batch, intermediate=(mode == "multiexit"))
        if mode == "teacher":
            prediction = _teacher_logits(runtime, head, output, batch, encoded)
            outputs = {32: prediction}
        else:
            outputs = _multi_logits(runtime, bank, teacher_head, output, batch, encoded, exit_layers)
        for layer in layers:
            all_logits[layer].extend(logits.detach().float().cpu() for logits in outputs[layer][0])
        all_targets.extend(targets)
        if index % 100 == 0:
            print(f"dev {index}/{len(rows)}", flush=True)
        del output, batch, encoded, outputs
    return {layer: runtime["summarize"](all_logits[layer], all_targets) for layer in layers}


def _train_stage(runtime, args, stage: str, output_dir: Path, teacher_state=None):
    torch = runtime["torch"]
    output_dir.mkdir(parents=True, exist_ok=True)
    exit_layers = runtime["exit_layers"]
    teacher_head = None
    if stage == "teacher":
        head = runtime["head_factory"]().to(device=runtime["device"], dtype=runtime["dtype"])
        trainable = list(head.parameters())
    else:
        teacher_head = runtime["head_factory"]().to(device=runtime["device"], dtype=runtime["dtype"])
        teacher_head.load_state_dict(teacher_state, strict=True)
        teacher_head.eval()
        for parameter in teacher_head.parameters():
            parameter.requires_grad_(False)
        head = runtime["ExitHeadBank"](exit_layers, runtime["head_factory"]).to(
            device=runtime["device"], dtype=runtime["dtype"])
        head.initialize_from(teacher_state)
        trainable = list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    best_metric = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        if teacher_head is not None:
            teacher_head.eval()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for index, row in enumerate(runtime["train_rows"], 1):
            if time.monotonic() - args._started_at > args.max_runtime_seconds:
                raise TimeoutError("M16 max-runtime-seconds reached during training")
            encoded, batch, targets, types = _device_batch(runtime, row, args.max_length)
            output = _base_hidden(runtime, batch, intermediate=(stage == "multiexit"))
            if stage == "teacher":
                question_logits = _teacher_logits(runtime, head, output, batch, encoded)[0]
                question_losses = [runtime["supervised_loss"](
                    logits.float(), target, args.brier_weight, args.label_smoothing)
                    for logits, target in zip(question_logits, targets)]
                loss = torch.stack(question_losses).mean()
            else:
                outputs = _multi_logits(runtime, head, teacher_head, output, batch, encoded, exit_layers)
                loss, _per_exit = runtime["aggregate_loss"](
                    outputs, [targets], [types], teacher_layer=32,
                    brier_weight=args.brier_weight, label_smoothing=args.label_smoothing,
                    distill_weight=args.distill_weight, temperature=args.temperature,
                    ordinal_rps_weight=args.ordinal_rps_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at train record {index}")
            group_start = ((index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(runtime["train_rows"]) - group_start)
            (loss / group_size).backward()
            if index % args.gradient_accumulation == 0 or index == len(runtime["train_rows"]):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())
            if index % 25 == 0 or index == len(runtime["train_rows"]):
                print(f"stage={stage} epoch={epoch}/{args.max_epochs} train={index}/{len(runtime['train_rows'])} "
                      f"mean_loss={running_loss/index:.6f}", flush=True)
            del output, batch, encoded, loss
        metrics = _evaluate(
            runtime, args, runtime["dev_rows"], stage,
            head if stage == "teacher" else None,
            teacher_head=teacher_head, bank=head if stage == "multiexit" else None,
            exit_layers=exit_layers,
        )
        criterion = metrics[32]["mean_nll"] if stage == "teacher" else sum(
            metrics[layer]["mean_nll"] for layer in exit_layers) / len(exit_layers)
        row = {"epoch": epoch, "stage": stage, "train_mean_loss": running_loss / len(runtime["train_rows"]),
               "selection_metric": "dev_mean_nll", "selection_value": criterion, "dev": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if criterion < best_metric:
            best_metric = criterion
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale_epochs = 0
            _atomic_torch_save(torch, best_state, output_dir / ("teacher_head.pt" if stage == "teacher" else "exit_heads.pt"))
        else:
            stale_epochs += 1
        _atomic_json(output_dir / "training_history.json", {"epochs": history})
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No finite best {stage} checkpoint was produced")
    head.load_state_dict(best_state, strict=True)
    final_metrics = _evaluate(
        runtime, args, runtime["dev_rows"], stage,
        head if stage == "teacher" else None,
        teacher_head=teacher_head, bank=head if stage == "multiexit" else None,
        exit_layers=exit_layers,
    )
    _atomic_json(output_dir / "dev_metrics.json", {str(layer): value for layer, value in final_metrics.items()})
    _atomic_json(output_dir / "head_config.json", runtime["head_config"])
    metadata = {
        "stage": stage, "base_model": args.base_model, "revision": args.revision,
        "train_records": len(runtime["train_rows"]), "dev_records": len(runtime["dev_rows"]),
        "train_families": sorted({str(row.get("family")) for row in runtime["train_rows"]}),
        "dev_families": sorted({str(row.get("family")) for row in runtime["dev_rows"]}),
        "exit_layers": list(exit_layers), "max_length": args.max_length, "seed": args.seed,
        "best_dev_mean_nll": best_metric, "outer_test_used": False,
        "max_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    _atomic_json(output_dir / "run_config.json", metadata)
    print(f"stage={stage} complete; weights={output_dir}", flush=True)
    return best_state


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    args._started_at = time.monotonic()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = _load_runtime(args)
    torch = runtime["torch"]
    torch.cuda.reset_peak_memory_stats()
    print(f"M16 runtime ready: model={args.base_model}@{args.revision} "
          f"train={len(runtime['train_rows'])} dev={len(runtime['dev_rows'])} "
          f"exits={runtime['exit_layers']} device={runtime['device']} dtype={runtime['dtype']}", flush=True)
    teacher_state = None
    if args.stage in ("teacher", "full"):
        teacher_dir = args.output / "teacher" if args.stage == "full" else args.output
        teacher_state = _train_stage(runtime, args, "teacher", teacher_dir)
    if args.stage in ("multiexit", "full"):
        if args.stage == "multiexit":
            checkpoint = args.teacher_checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint}")
            teacher_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            shutil.copy2(checkpoint, args.output / "teacher_head.pt")
        multiexit_dir = args.output / "multiexit" if args.stage == "full" else args.output
        _train_stage(runtime, args, "multiexit", multiexit_dir, teacher_state=teacher_state)
    stage_names = ("teacher",) if args.stage == "teacher" else (
        ("multiexit",) if args.stage == "multiexit" else ("teacher", "multiexit")
    )
    stage_metrics = {}
    for stage_name in stage_names:
        stage_dir = args.output / stage_name if args.stage == "full" else args.output
        stage_metrics[stage_name] = json.loads((stage_dir / "dev_metrics.json").read_text(encoding="utf-8"))
    _atomic_json(args.output / "run_report.json", {
        "status": "completed", "stage": args.stage, "stage_metrics": stage_metrics,
        "train_records": len(runtime["train_rows"]), "dev_records": len(runtime["dev_rows"]),
        "exit_layers": list(runtime["exit_layers"]), "outer_test_used": False,
    })
    print("M16 training stages completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
