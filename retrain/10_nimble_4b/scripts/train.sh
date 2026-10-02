#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(cd ../.. && pwd)"
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then
  echo "This starts training. After reviewing configs/train_4b.yaml, rerun with CONFIRM_TRAIN=yes." >&2
  exit 2
fi
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python -m nimble.training.schema_train train \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --data-dir data/raw \
  --validation data/raw/eval.jsonl \
  --output-dir weights/nimble-4b \
  --learning-rate 5e-5 \
  --lora-rank 16 \
  --seed 17 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --max-length 2048 \
  --max-steps 1005 \
  --warmup-steps 101 \
  --save-steps 335 \
  --stop-after-epochs 1
