#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f weights/full_ft/head.pt ]] || { echo "Checkpoint missing: run train.sh with CONFIRM_TRAIN=yes first." >&2; exit 2; }
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python -m kev.benchmark \
  --run weights/full_ft \
  --data ../10_nimble_4b/data/processed/kev_eval.jsonl \
  --out results/eval
