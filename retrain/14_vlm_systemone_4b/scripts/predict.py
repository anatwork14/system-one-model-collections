#!/usr/bin/env python3
"""Score one explicitly selected M14 split with a saved V1 checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration

from train import DATA, METHOD, MODEL_PATH, MODEL_REVISION, candidate_features, file_hash, load_processor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("model_validation", "calibration", "test", "ood"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("M14 prediction requires a compatible CUDA PyTorch runtime")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint is missing: {args.checkpoint}")
    manifest_path = DATA / "manifests/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_path = DATA / "manifests/leakage_audit.json"
    if manifest.get("status") != "ready" or not audit_path.is_file():
        raise SystemExit("M14 dataset and full leakage audit must be ready before scoring")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("manifest_sha256") != file_hash(manifest_path):
        raise SystemExit("Dataset leakage audit is stale or failed")
    split_spec = manifest.get("splits", {}).get(args.split)
    if not split_spec:
        raise SystemExit(f"Requested split is missing: {args.split}")
    split_path = DATA / split_spec["path"]
    if file_hash(split_path) != split_spec.get("sha256") or audit["split_hashes"].get(args.split) != split_spec.get("sha256"):
        raise SystemExit(f"Split hash/audit mismatch for {args.split}")
    rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Requested split is empty: {args.split}")
    for row in rows:
        for relative in row.get("images", []):
            image_path = (DATA / relative).resolve()
            audited = audit.get("image_files", {}).get(relative)
            try:
                image_path.relative_to(DATA.resolve())
                image_inside_data = True
            except ValueError:
                image_inside_data = False
            if (not image_inside_data or not image_path.is_file() or
                    not audited or file_hash(image_path) != audited.get("sha256")):
                raise SystemExit(f"Scoring image is missing, outside the dataset, or changed since the full audit: {relative}")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("model_revision") != MODEL_REVISION:
        raise SystemExit("Checkpoint was trained against a different model revision")
    if state.get("data_hashes", {}).get("manifest") != file_hash(manifest_path):
        raise SystemExit("Checkpoint was trained against a different M14 manifest")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = load_processor(int(state.get("config", {}).get("max_image_pixels", 786432)))
    max_sequence_tokens = int(state.get("config", {}).get("max_sequence_tokens", 2048))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = nn.Linear(int(model.config.text_config.hidden_size), 1, dtype=torch.float32)
    head.load_state_dict(state["head"])
    head = head.to("cuda").eval()
    checkpoint_sha = file_hash(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = correct = 0
    nll_sum = brier_sum = 0.0
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            features = candidate_features(model, processor, row, torch.device("cuda"), dtype, max_sequence_tokens)
            logits = head(features).squeeze(-1).float()
            probabilities = logits.softmax(dim=-1)
            target_index = next(i for i, item in enumerate(row["candidates"]) if item["id"] == row["target"])
            target = torch.tensor([target_index], device="cuda")
            nll = float(nn.functional.cross_entropy(logits.unsqueeze(0), target))
            one_hot = nn.functional.one_hot(target[0], num_classes=len(row["candidates"])).float()
            brier = float((probabilities - one_hot).square().sum())
            candidate_ids = [item["id"] for item in row["candidates"]]
            prediction = candidate_ids[int(logits.argmax())]
            torch.cuda.synchronize()
            result = {"id": row["id"], "split": args.split, "target": row["target"],
                      "predicted": prediction,
                      "probabilities": {cid: float(probabilities[i]) for i, cid in enumerate(candidate_ids)},
                      "latency_ms": (time.perf_counter() - started) * 1000.0,
                      "nll": nll, "brier": brier, "checkpoint_sha256": checkpoint_sha,
                      "manifest_sha256": file_hash(manifest_path), "split_sha256": split_spec["sha256"]}
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
            correct += int(prediction == row["target"])
            nll_sum += nll
            brier_sum += brier
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(json.dumps({"event": "prediction_progress", "split": args.split,
                                  "done": index, "total": len(rows)}, flush=True))
    summary = {"split": args.split, "questions": count, "accuracy": correct / count,
               "nll": nll_sum / count, "brier": brier_sum / count,
               "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
               "predictions": str(args.output)}
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
