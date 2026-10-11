#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for data_file in data/processed/train_pairs_inner.jsonl data/processed/dev_pairs_inner.jsonl; do
  [[ -s "$data_file" ]] || { echo "Missing clean inner-split input $data_file; run scripts/prepare_data.sh before packaging this VKong task." >&2; exit 2; }
done

python3 scripts/train_contrastive.py \
  --config configs/clm.yaml \
  --train data/processed/train_pairs_inner.jsonl \
  --validation data/processed/dev_pairs_inner.jsonl \
  --base Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output /data/checkpoints/clm-heads-clean \
  --epochs 20 \
  --learning-rate 1e-4 \
  --projection-dim 512 \
  --device cuda
