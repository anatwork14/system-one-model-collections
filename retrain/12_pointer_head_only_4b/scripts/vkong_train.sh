#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for data_file in data/processed/kev_train_inner.jsonl data/processed/kev_dev_inner.jsonl; do
  [[ -s "$data_file" ]] || { echo "Missing clean inner-split input $data_file; run the local preparation script before packaging this VKong task." >&2; exit 2; }
done

export PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 scripts/train_head_only.py \
  --upstream upstream \
  --train data/processed/kev_train_inner.jsonl \
  --validation data/processed/kev_dev_inner.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --output /data/checkpoints/pointer-head-only-clean \
  --epochs 20 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --seed 17 \
  --head-dim 256 \
  --max-state 2048 \
  --patience 5 \
  --device cuda
