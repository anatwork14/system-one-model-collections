#!/usr/bin/env python3
"""Run P0: extract frozen layer features, fit identical probes, and score inner-dev."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent


def find_workspace_root() -> Path:
    """Resolve the uploaded checkout from the probe path or its working directory."""
    marker = Path("retrain/16_adas1_multiexit_4b/configs/probe.yaml")
    for start in (HERE, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / marker).is_file():
                return candidate
    raise FileNotFoundError(
        "cannot locate uploaded M16 workspace; expected "
        "retrain/16_adas1_multiexit_4b/configs/probe.yaml"
    )


ROOT = find_workspace_root()
RELEASE_ARCH = ROOT / "released/05_clef_flash_9b/weights/architecture"
HELPER = Path(os.environ.get(
    "M16_MODEL_HELPER", str(RELEASE_ARCH / "joint_schema_model.py")
)).resolve()
if not HELPER.is_file():
    # Some source uploaders stage an explicitly listed file at the project root.
    staged_helper = ROOT / "joint_schema_model.py"
    if staged_helper.is_file():
        HELPER = staged_helper
    else:
        raise FileNotFoundError(
            "M16 requires joint_schema_model.py; checked "
            f"{RELEASE_ARCH / 'joint_schema_model.py'} and {staged_helper}"
        )
sys.path.insert(0, str(HELPER.parent))
sys.path.insert(0, str(METHOD / "src"))
sys.path.insert(0, str(HERE))
from joint_schema_model import collate_records, encode_record  # noqa: E402
from adas1.probe import LayerProbe, objective  # noqa: E402
from vkong_m16_service import validate_layers_for_model  # noqa: E402

LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
CONTROLLED_DATA = METHOD / "data/controlled"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_index(question: dict, options: tuple[str, ...]) -> int:
    target = question["target"]
    target_id = ("true" if target else "false") if question["type"] == "noul" else str(target)
    try:
        return options.index(target_id)
    except ValueError as exc:
        raise ValueError(f"Gold option {target_id!r} missing from schema") from exc


def cache_split(rows: list[dict], split: str, tokenizer, backbone, device, max_length: int,
                cache_dir: Path, replace: bool = False) -> None:
    path = cache_dir / f"{split}_features.pt"
    if path.is_file() and not replace:
        return
    text_model = backbone.model
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    cached = {layer: [] for layer in LAYERS}
    backbone.eval()
    with torch.inference_mode():
        for row_index, row in enumerate(rows, 1):
            encoded = encode_record(tokenizer, row, max_length=max_length)
            batch = collate_records([encoded], tokenizer.pad_token_id or 0, device)
            output = text_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                use_cache=False, output_hidden_states=True, return_dict=True,
            )
            valid_length = int(batch["attention_mask"][0].sum().item())
            record_layers = {layer: [] for layer in LAYERS}
            for layer in LAYERS:
                hidden = output.hidden_states[layer][0, :valid_length]
                global_state = hidden[-1].detach().to(device="cpu", dtype=torch.float16)
                for encoded_question, question in zip(encoded.questions, row["questions"].values()):
                    q_start, q_end = encoded_question.question_span
                    q_vector = hidden[q_start:q_end].mean(dim=0).detach().to(device="cpu", dtype=torch.float16)
                    option_vectors = [hidden[start:end].mean(dim=0).detach().to(device="cpu", dtype=torch.float16)
                                      for start, end in encoded_question.option_spans]
                    record_layers[layer].append({
                        "question": q_vector,
                        "global": global_state,
                        "options": torch.stack(option_vectors),
                        "target": target_index(question, encoded_question.option_ids),
                        "type": question["type"],
                        "option_ids": list(encoded_question.option_ids),
                    })
                del hidden
            for layer in LAYERS:
                cached[layer].append({"id": row["id"], "questions": record_layers[layer]})
            if row_index % 25 == 0 or row_index == len(rows):
                print(f"{split}: extracted {row_index}/{len(rows)} records", flush=True)
            del output, batch
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"layers": cached, "record_ids": [row["id"] for row in rows]}, temporary)
    temporary.replace(path)


def summarize(model, examples, device) -> tuple[dict, float]:
    model.eval()
    correct = count = 0
    nll_total = brier_total = 0.0
    calibration = [[] for _ in range(15)]
    with torch.inference_mode():
        for record in examples:
            for example in record["questions"]:
                logits = model(example["question"].to(device=device, dtype=torch.float32),
                               example["global"].to(device=device, dtype=torch.float32),
                               example["options"].to(device=device, dtype=torch.float32))
                probs = logits.softmax(dim=-1)
                target = int(example["target"])
                pred = int(probs.argmax().item())
                confidence = float(probs[pred].item())
                is_correct = pred == target
                nll_total += -math.log(max(float(probs[target].item()), 1e-12))
                brier_total += float((probs - torch.nn.functional.one_hot(
                    torch.tensor(target, device=device), num_classes=probs.numel()
                ).to(probs.dtype)).square().sum().item())
                correct += int(is_correct)
                count += 1
                calibration[min(14, int(confidence * 15))].append((confidence, is_correct))
    ece = sum(len(bucket) / max(count, 1) * abs(
        sum(item[0] for item in bucket) / len(bucket) -
        sum(item[1] for item in bucket) / len(bucket)
    ) for bucket in calibration if bucket)
    metrics = {"questions": count, "accuracy": correct / max(count, 1),
               "nll": nll_total / max(count, 1), "brier": brier_total / max(count, 1), "ece_15": ece}
    return metrics, metrics["nll"]


def fit_probe(layer: int, train: list[dict], dev: list[dict], args, device, output_dir: Path) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = LayerProbe(projection_dim=args.projection_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_nll, best_state, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        train_loss = 0.0
        questions_seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, record_index in enumerate(order):
            record = train[record_index]
            terms = []
            for example in record["questions"]:
                logits = model(example["question"].to(device=device, dtype=torch.float32),
                               example["global"].to(device=device, dtype=torch.float32),
                               example["options"].to(device=device, dtype=torch.float32))
                terms.append(objective(logits, int(example["target"]), args.brier_weight))
            if not terms:
                continue
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite P0 loss at layer {layer}, record {record['id']}")
            group_start = (batch_index // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, len(order) - group_start)
            (loss / group_size).backward()
            train_loss += float(loss.detach())
            questions_seen += len(terms)
            if (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        dev_metrics, dev_nll = summarize(model, dev, device)
        epoch_row = {"epoch": epoch + 1, "train_mean_record_loss": train_loss / max(len(train), 1),
                     "train_questions": questions_seen, "dev": dev_metrics}
        history.append(epoch_row)
        print(json.dumps({"layer": layer, **epoch_row}), flush=True)
        if dev_nll < best_nll:
            best_nll, best_epoch, stale = dev_nll, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"no valid checkpoint selected for layer {layer}")
    model.load_state_dict(best_state)
    metrics, _ = summarize(model, dev, device)
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, layer_dir / "probe.pt")
    (layer_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {"layer": layer, "fraction_of_full_decoder": layer / 32, "best_epoch": best_epoch, **metrics}


def write_curve_svg(rows: list[dict], path: Path) -> None:
    """Write a dependency-free four-panel plot of the inner-dev exit curve."""
    width, panel_height, left, right = 860, 190, 74, 24
    height = panel_height * 4 + 30
    panels = [("Accuracy", "accuracy", False), ("NLL", "nll", True),
              ("Brier", "brier", True), ("ECE (15 bins)", "ece_15", True)]
    colors = {"accuracy": "#2563eb", "nll": "#dc2626", "brier": "#059669", "ece_15": "#7c3aed"}
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font:13px sans-serif;fill:#1f2937}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}</style>']
    for panel_index, (title, key, lower_is_better) in enumerate(panels):
        top = panel_index * panel_height + 20
        plot_top, plot_bottom = top + 28, top + 150
        values = [float(row[key]) for row in rows]
        low, high = min(values), max(values)
        if high - low < 1e-9:
            low, high = max(0.0, low - 0.05), high + 0.05
        padding = (high - low) * 0.15
        low, high = max(0.0, low - padding), high + padding
        pieces.append(f'<text x="{left}" y="{top + 15}" font-weight="bold">{title} (dev; lower is better: {str(lower_is_better).lower()})</text>')
        for tick in range(3):
            y = plot_bottom - tick * (plot_bottom - plot_top) / 2
            value = low + tick * (high - low) / 2
            pieces.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
            pieces.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{value:.3f}</text>')
        points = []
        for index, row in enumerate(rows):
            x = left + index * (width - left - right) / max(len(rows) - 1, 1)
            y = plot_bottom - (float(row[key]) - low) * (plot_bottom - plot_top) / (high - low)
            points.append(f"{x:.1f},{y:.1f}")
            pieces.append(f'<text x="{x:.1f}" y="{plot_bottom+19}" text-anchor="middle">{row["layer"]}</text>')
        pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[key]}" stroke-width="2.5"/>')
        for point in points:
            x, y = point.split(",")
            pieces.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[key]}"/>')
        pieces.append(f'<line class="axis" x1="{left}" y1="{plot_bottom}" x2="{width-right}" y2="{plot_bottom}"/>')
    pieces.append("</svg>")
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def main() -> None:
    global LAYERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(ROOT / "shared/models/qwen3.5-4b"))
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)),
                        help="comma-separated decoder layer indices (normally supplied from configs/probe.yaml)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--brier-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--replace-features", action="store_true")
    args = parser.parse_args()
    try:
        LAYERS = tuple(int(value) for value in args.layers.split(","))
    except ValueError as exc:
        raise SystemExit("--layers must be a comma-separated list of integers") from exc
    if not LAYERS or tuple(sorted(set(LAYERS))) != LAYERS or any(layer < 1 or layer > 32 for layer in LAYERS):
        raise SystemExit(f"--layers must be unique increasing values from 1 to 32; got {LAYERS}")
    if not torch.cuda.is_available():
        raise SystemExit("P0 requires a compatible CUDA runtime; this script will not fall back to slow CPU extraction.")
    gpu_properties = torch.cuda.get_device_properties(0)
    free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(0)
    print(json.dumps({
        "memory_fit_check": "actual model load and feature extraction",
        "gpu": gpu_properties.name,
        "gpu_free_memory_bytes": free_memory_bytes,
        "gpu_total_memory_bytes": total_memory_bytes,
    }), flush=True)
    missing_kernels = [name for name in ("fla", "causal_conv1d") if importlib.util.find_spec(name) is None]
    if missing_kernels:
        raise SystemExit("Install Qwen3.5 optimized kernels before P0; missing imports: " + ", ".join(missing_kernels))
    train_path, dev_path = CONTROLLED_DATA / "train.jsonl", CONTROLLED_DATA / "dev.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        raise SystemExit("Run scripts/prepare_data.py first; P0 only accepts its explicit train/dev files.")
    train_rows, dev_rows = read_jsonl(train_path), read_jsonl(dev_path)
    train_ids, dev_ids = {row["id"] for row in train_rows}, {row["id"] for row in dev_rows}
    train_families, dev_families = {row["family"] for row in train_rows}, {row["family"] for row in dev_rows}
    train_sources, dev_sources = {row["source_id"] for row in train_rows}, {row["source_id"] for row in dev_rows}
    if train_ids & dev_ids or train_families & dev_families or train_sources & dev_sources:
        raise ValueError("P0 data leak: train and dev overlap by record, family, or source id")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or METHOD / "results/probes" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = output / "features"
    cache_dir.mkdir()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                      "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": str(dtype)}, indent=2), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    model_config = backbone.config
    text_config = getattr(model_config, "text_config", None)
    actual_layers = getattr(text_config, "num_hidden_layers", None)
    if actual_layers is None:
        actual_layers = getattr(model_config, "num_hidden_layers", None)
    if not isinstance(actual_layers, int) or actual_layers < 1:
        raise ValueError("loaded model config does not expose a valid decoder num_hidden_layers")
    validate_layers_for_model(LAYERS, actual_layers)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    cache_split(train_rows, "train", tokenizer, backbone, device, args.max_length, cache_dir, args.replace_features)
    cache_split(dev_rows, "dev", tokenizer, backbone, device, args.max_length, cache_dir, args.replace_features)
    del backbone
    torch.cuda.empty_cache()
    train_cache = torch.load(cache_dir / "train_features.pt", map_location="cpu", weights_only=False)["layers"]
    dev_cache = torch.load(cache_dir / "dev_features.pt", map_location="cpu", weights_only=False)["layers"]
    rows = []
    for layer in LAYERS:
        metrics = fit_probe(layer, train_cache[layer], dev_cache[layer], args, device, output)
        rows.append(metrics)
    with (output / "exit_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_curve_svg(rows, output / "exit_curve.svg")
    report = {
        "method": "M16 AdaS1 P0 frozen layer probe",
        "status": "probe_complete_exit_selection_pending_review",
        "base_model": args.base_model,
        "base_revision": args.revision,
        "layers": list(LAYERS),
        "train_records": len(train_rows), "dev_records": len(dev_rows),
        "train_families": len(train_families), "dev_families": len(dev_families),
        "outer_324_test_loaded": False,
        "runtime": {"gpu": torch.cuda.get_device_name(0), "gpu_total_memory_bytes": gpu_properties.total_memory,
                    "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "precision": str(dtype),
                    "elapsed_seconds": time.perf_counter() - run_started,
                    "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
        "probe": {"projection_dim": args.projection_dim, "hidden_dim": args.hidden_dim,
                  "brier_weight": args.brier_weight,
                  "objective": f"cross-entropy + {args.brier_weight} * multiclass Brier",
                  "selection": "inner-dev mean question NLL"},
        "results": rows,
        "limitations": ["Synthetic/model-checked Nimble labels; P0 is an architecture screen, not a benchmark claim.",
                        "Probe performance does not demonstrate early-exit serving speed."],
    }
    (output / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(output / "probe_report.json"), "probe_report": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
