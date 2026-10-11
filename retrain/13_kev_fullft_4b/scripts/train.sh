#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
[[ ! -e weights/full_ft-clean ]] || { echo "Output already exists: weights/full_ft-clean. Choose a new run directory; checkpoints are never overwritten." >&2; exit 2; }
ROOT="$(cd ../.. && pwd)"
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
python3 ../10_nimble_4b/scripts/prepare_inner_split.py \
  --source-dir ../10_nimble_4b/data/raw \
  --output-dir ../10_nimble_4b/data/inner_split \
  --seed 17
python3 scripts/verify_training_data.py
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python3 -m kev.train \
  --base Qwen/Qwen3.5-4B-Base \
  --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --data data/processed/kev_train_inner.jsonl \
  --out weights/full_ft-clean \
  --epochs 1 --lr 1e-5 --head_lr 1e-4 --batch 1 --accum 8 --seed 17 \
  --dtype bf16 --weights_dtype bf16 --checkpointing 1 --full_ft 1 --length_sort 1
