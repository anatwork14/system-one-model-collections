#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:-weights/head_only-clean}"
[[ -f "$checkpoint/pointer_head.pt" ]] || { echo "Checkpoint missing: $checkpoint" >&2; exit 2; }
python3 scripts/infer.py --checkpoint "$checkpoint" --input data/processed/kev_dev_inner.jsonl --output results/dev_predictions.jsonl
python3 ../../evaluation/standardize_training_eval.py M12 \
  --native results/dev_predictions.jsonl --source data/processed/kev_dev_inner.jsonl \
  --output results/dev_metric_predictions.jsonl
python3 ../../evaluation/metrics/compute_metrics.py results/dev_metric_predictions.jsonl --output results/dev_metrics.json
