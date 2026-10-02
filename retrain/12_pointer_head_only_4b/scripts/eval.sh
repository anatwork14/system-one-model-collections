#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f weights/head_only/pointer_head.pt ]] || { echo "Checkpoint missing: run train.sh with CONFIRM_TRAIN=yes first." >&2; exit 2; }
python scripts/infer.py --input ../10_nimble_4b/data/processed/kev_eval.jsonl --output results/eval.jsonl
python ../../evaluation/metrics/compute_metrics.py results/eval.jsonl > results/metrics.json
