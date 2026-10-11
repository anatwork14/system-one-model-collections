#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d weights/nimble-4b-clean ]]; then echo "Clean checkpoint missing: run scripts/train.sh first." >&2; exit 2; fi
export HF_HOME="$(cd ../.. && pwd)/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
python3 scripts/eval_nimble.py
python3 ../../evaluation/standardize_training_eval.py M10 \
  --native results/dev.json --output results/dev_metric_predictions.jsonl
python3 ../../evaluation/metrics/compute_metrics.py results/dev_metric_predictions.jsonl --output results/dev_standard_metrics.json
