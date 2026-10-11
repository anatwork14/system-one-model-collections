#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:-weights/clm-heads-clean}"
[[ -f "$checkpoint/heads.pt" ]] || { echo "CLM checkpoint missing: $checkpoint" >&2; exit 2; }
python3 scripts/infer.py --config configs/clm.yaml --checkpoint "$checkpoint" \
  --input data/processed/dev_pairs_inner.jsonl --output results/dev_predictions.jsonl
python3 ../../evaluation/standardize_training_eval.py M11 \
  --native results/dev_predictions.jsonl --output results/dev_metric_predictions.jsonl
python3 ../../evaluation/metrics/compute_metrics.py results/dev_metric_predictions.jsonl --output results/dev_metrics.json
