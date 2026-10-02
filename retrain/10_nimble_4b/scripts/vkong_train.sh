#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/upstream${PYTHONPATH:+:${PYTHONPATH}}"

python -m nimble.training.schema_train train \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --data-dir data/raw \
  --validation data/raw/eval.jsonl \
  --output-dir /data/checkpoints/nimble-4b \
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
