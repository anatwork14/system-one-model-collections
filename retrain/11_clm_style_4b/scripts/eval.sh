#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f weights/clm-heads/heads.pt ]]; then echo "CLM head checkpoint missing; run train.sh with CONFIRM_TRAIN=yes first." >&2; exit 2; fi
python scripts/infer.py --config configs/clm.yaml --input ../10_nimble_4b/data/processed/eval_pairs.jsonl --output results/eval.jsonl
