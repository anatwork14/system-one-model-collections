#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then echo "Training is disabled unless CONFIRM_TRAIN=yes is set." >&2; exit 2; fi
ROOT="$(cd ../.. && pwd)"
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
python3 ../10_nimble_4b/scripts/prepare_inner_split.py \
  --source-dir ../10_nimble_4b/data/raw \
  --output-dir ../10_nimble_4b/data/inner_split \
  --seed 17
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python3 scripts/train_head_only.py \
  --upstream upstream \
  --train data/processed/kev_train_inner.jsonl \
  --validation data/processed/kev_dev_inner.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --base-revision 1001bb4d826a52d1f399e183466143f4da7b741b \
  --output weights/head_only-clean \
  --epochs 20 --learning-rate 1e-4 --weight-decay 0.01 --seed 17 --patience 5
