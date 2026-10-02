#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
ROOT="$(cd ../.. && pwd)"
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python -m kev.train \
  --base Qwen/Qwen3.5-4B-Base \
  --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --data ../10_nimble_4b/data/processed/kev_train.jsonl \
  --out weights/full_ft \
  --epochs 1 --lr 1e-5 --head_lr 1e-4 --batch 1 --accum 8 --seed 17 \
  --dtype bf16 --weights_dtype bf16 --checkpointing 1 --full_ft 1
