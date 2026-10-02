#!/usr/bin/env python3
"""Compute choice metrics from JSONL rows with `label` and `probabilities` fields."""
import argparse
import json
import math
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--bins", type=int, default=15)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.predictions.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("No prediction rows")
    acc = nll = brier = 0.0
    conf_corr = []
    risks = []
    for row in rows:
        probs = row["probabilities"]
        y = str(row["label"])
        if y not in probs:
            raise ValueError(f"label {y!r} missing from probabilities for {row.get('id')}")
        pred = max(probs, key=probs.get)
        correct = float(pred == y)
        confidence = float(probs[pred])
        acc += correct
        nll -= math.log(max(float(probs[y]), 1e-15))
        brier += sum((float(v) - float(k == y)) ** 2 for k, v in probs.items())
        conf_corr.append((confidence, correct))
        risks.append((confidence, 1.0 - correct))
    n = len(rows)
    ece = 0.0
    for i in range(a.bins):
        lo, hi = i / a.bins, (i + 1) / a.bins
        bucket = [(c, ok) for c, ok in conf_corr if lo <= c < hi or (i == a.bins - 1 and c == 1)]
        if bucket:
            ece += len(bucket) / n * abs(sum(x for x, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket))
    risks.sort(reverse=True)
    # Area under the risk-coverage curve, using confidence-ranked predictions.
    cum = 0.0
    aurc = 0.0
    for i, (_, risk) in enumerate(risks, 1):
        cum += risk
        aurc += cum / i
    aurc /= n
    print(json.dumps({"n": n, "accuracy": acc / n, "nll": nll / n, "brier": brier / n,
                      "ece": ece, "aurc": aurc, "ece_bins": a.bins}, indent=2))

if __name__ == "__main__": main()
