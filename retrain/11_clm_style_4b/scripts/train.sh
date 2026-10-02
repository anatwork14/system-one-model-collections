#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
bash scripts/prepare_data.sh
python scripts/train_contrastive.py --config configs/clm.yaml \
  --train data/processed/train_pairs.jsonl \
  --validation data/processed/eval_pairs.jsonl \
  --base weights/base_model --output weights/clm-heads
