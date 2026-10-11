#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

train_data="data/processed/kev_train_inner.jsonl"
[[ -s "$train_data" ]] || { echo "Missing clean inner-split training data: $train_data. Run scripts/train.sh locally once to prepare method data before packaging." >&2; exit 2; }
python3 scripts/verify_training_data.py
output="/data/checkpoints/kev-fullft-clean"
[[ ! -e "$output" ]] || { echo "Output already exists: $output. Choose a new run directory; this launcher never deletes checkpoints." >&2; exit 2; }
mkdir -p "$(dirname "$output")"

export PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m kev.train \
  --base Qwen/Qwen3.5-4B-Base \
  --base_revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --data "$train_data" \
  --out "$output" \
  --epochs 1 --lr 1e-5 --head_lr 1e-4 --batch 1 --accum 8 --seed 17 \
  --device cuda --dtype bf16 --weights_dtype bf16 --checkpointing 1 --full_ft 1 --length_sort 1
