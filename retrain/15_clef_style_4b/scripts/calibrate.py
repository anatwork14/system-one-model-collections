#!/usr/bin/env python3
"""Fit one scalar temperature using saved calibration-split logits."""

import argparse
import json
import math
from pathlib import Path


def nll(rows, temperature):
    total = 0.0
    for row in rows:
        values = [float(value) / temperature for value in row["logits"]]
        peak = max(values)
        log_z = peak + math.log(sum(math.exp(value - peak) for value in values))
        target = row["option_ids"].index(str(row["target"]))
        total += log_z - values[target]
    return total / max(1, len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Calibration JSONL with option_ids, target, logits")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/temperature.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    candidates = [0.25 + index * 0.025 for index in range(791)]
    temperature = min(candidates, key=lambda value: nll(rows, value))
    report = {"method": "global_temperature", "temperature": temperature, "calibration_nll": nll(rows, temperature), "examples": len(rows), "fit_split": "calibration_only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
