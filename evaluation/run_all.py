#!/usr/bin/env python3
"""Inventory methods and validate normalized probability records."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "released/00_semif", "released/01_simple_jev", "released/02_tev1_4b",
    "released/03_jevk5_4b", "released/04_kev_4b", "released/05_clef_flash_9b",
    "retrain/10_nimble_4b",
    "retrain/11_clm_style_4b", "retrain/12_pointer_head_only_4b",
    "retrain/13_kev_fullft_4b", "retrain/14_vlm_systemone_4b",
    "retrain/15_clef_style_4b",
]
DEFERRED = {"retrain/14_vlm_systemone_4b"}
FIELDS = ["method", "base_model", "checkpoint_revision", "dataset", "split", "id", "label",
          "input_tokens", "num_options", "predicted_option", "probabilities", "latency_ms",
          "router_latency_ms", "gpu_seconds", "peak_vram_mb", "cost_usd", "inference_success",
          "task_success", "served_locally", "external_offload", "escalated", "notes"]

def validate(record):
    for field in ("id", "method", "base_model", "checkpoint_revision", "dataset", "split", "label",
                  "input_tokens", "num_options", "predicted_option", "probabilities", "latency_ms"):
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"{field} is required")
    for field, minimum in (("input_tokens", 0), ("num_options", 2)):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    latency = record["latency_ms"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError("latency_ms must be finite numeric and nonnegative")
    probs = record.get("probabilities")
    if not isinstance(probs, dict) or len(probs) < 2:
        raise ValueError("probabilities must map at least two option IDs to values")
    vals = list(probs.values())
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vals):
        raise ValueError("probabilities must be numeric")
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in vals) or abs(sum(vals) - 1.0) > 1e-4:
        raise ValueError("probability values must be in [0,1] and sum to 1 within 1e-4")
    if len(probs) != record["num_options"]:
        raise ValueError("num_options must match the probability map size")
    label = "true" if record["label"] is True else "false" if record["label"] is False else str(record["label"])
    if label not in probs:
        raise ValueError("label is missing from probabilities")
    predicted = max(probs, key=probs.get)
    if str(record.get("predicted_option", predicted)) != predicted:
        raise ValueError("predicted_option must match the probability argmax")
    if "inference_success" in record and not isinstance(record["inference_success"], bool):
        raise ValueError("inference_success must be boolean")
    if "task_success" in record:
        success = record["task_success"]
        if isinstance(success, bool) or not isinstance(success, (int, float)) or not math.isfinite(success) or not 0 <= success <= 1:
            raise ValueError("task_success must be finite numeric and in [0,1]")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list method directories")
    parser.add_argument("--validate-jsonl", type=Path, help="validate normalized result JSONL")
    args = parser.parse_args()
    if args.validate_jsonl:
        count = 0
        seen_ids = set()
        with args.validate_jsonl.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    record = json.loads(line)
                    validate(record)
                    if record["id"] in seen_ids:
                        raise ValueError(f"duplicate prediction id on line {line_no}: {record['id']}")
                    seen_ids.add(record["id"])
                    count += 1
        if not count:
            raise ValueError(f"No prediction rows in {args.validate_jsonl}")
        print(f"valid records: {count}")
    elif args.list:
        for method in METHODS:
            status = "deferred" if method in DEFERRED else "documented" if (ROOT / method / 'METHOD.md').exists() else "missing"
            print(f"{status}\t{method}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
