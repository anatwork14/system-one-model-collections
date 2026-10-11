#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
bash scripts/prepare_data.sh
python3 scripts/train_contrastive.py --config configs/clm.yaml \
  --train data/processed/train_pairs_inner.jsonl \
  --validation data/processed/dev_pairs_inner.jsonl \
  --base Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output weights/clm-heads-clean --epochs 20 --learning-rate 1e-4 --projection-dim 512
