#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
ROOT="$(cd ../.. && pwd)"
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python scripts/train_head_only.py \
  --upstream upstream \
  --train ../10_nimble_4b/data/processed/kev_train.jsonl \
  --validation ../10_nimble_4b/data/processed/kev_eval.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --output weights/head_only \
  --epochs 20 --learning-rate 1e-3 --weight-decay 0.01 --seed 17
