#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
checkpoint="${1:-weights/full_ft-clean}"
output="${2:-${KEV_EVAL_OUT:-results/dev}}"
[[ -f "$checkpoint/head.pt" ]] || { echo "Checkpoint missing: $checkpoint" >&2; exit 2; }
python3 scripts/verify_training_data.py
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python3 -m kev.benchmark \
  --run "$checkpoint" \
  --data data/processed/kev_dev_inner.jsonl \
  --out "$output"
