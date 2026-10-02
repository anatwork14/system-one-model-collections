#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/train_contrastive.py \
  --config configs/clm.yaml \
  --train data/processed/train_pairs.jsonl \
  --validation data/processed/eval_pairs.jsonl \
  --base Qwen/Qwen3.5-4B \
  --output /data/checkpoints/clm-heads \
  --epochs 20 \
  --learning-rate 1e-3 \
  --projection-dim 512 \
  --device cuda
