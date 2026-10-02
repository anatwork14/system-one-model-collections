#!/usr/bin/env python3
"""Inventory methods and validate normalized probability records."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "released/00_semif", "released/01_simple_jev", "released/02_tev1_4b",
    "released/03_jevk5_4b", "released/04_kev_4b", "retrain/10_nimble_4b",
    "retrain/11_clm_style_4b", "retrain/12_pointer_head_only_4b",
    "retrain/13_kev_fullft_4b", "retrain/14_vlm_systemone_4b",
]
FIELDS = ["method", "base_model", "checkpoint_revision", "input_tokens", "num_options", "predicted_option", "probabilities", "latency_ms", "peak_vram_mb", "success", "notes"]

def validate(record):
    probs = record.get("probabilities")
    if not isinstance(probs, dict) or len(probs) < 2:
        raise ValueError("probabilities must map at least two option IDs to values")
    vals = list(probs.values())
    if any(not isinstance(x, (int, float)) for x in vals):
        raise ValueError("probabilities must be numeric")
    if any(x < 0 or x > 1 for x in vals) or abs(sum(vals) - 1.0) > 1e-4:
        raise ValueError("probability values must be in [0,1] and sum to 1 within 1e-4")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list method directories")
    parser.add_argument("--validate-jsonl", type=Path, help="validate normalized result JSONL")
    args = parser.parse_args()
    if args.validate_jsonl:
        count = 0
        with args.validate_jsonl.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    validate(json.loads(line)); count += 1
        print(f"valid records: {count}")
    elif args.list:
        for method in METHODS:
            print(f"{'ready' if (ROOT / method / 'METHOD.md').exists() else 'missing'}\t{method}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
