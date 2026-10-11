#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
export PYTHONPATH="${ROOT}/upstream${PYTHONPATH:+:${PYTHONPATH}}"
python3 scripts/prepare_inner_split.py --source-dir data/raw --output-dir data/inner_split --seed 17
steps_per_epoch="$(python3 -c 'import math; n=sum(1 for line in open("data/inner_split/train.jsonl", encoding="utf-8") if line.strip()); print(math.ceil(math.ceil(n / 2) / 4))')"
warmup_steps="$((steps_per_epoch / 10))"
if [[ "$warmup_steps" -lt 1 ]]; then warmup_steps=1; fi

python3 -m nimble.training.schema_train train \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --data-dir data/inner_split \
  --validation data/inner_split/validation.jsonl \
  --output-dir /data/checkpoints/nimble-4b-clean \
  --learning-rate 5e-5 \
  --lora-rank 16 \
  --seed 17 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --max-length 2048 \
  --max-steps "$steps_per_epoch" \
  --warmup-steps "$warmup_steps" \
  --save-steps "$steps_per_epoch" \
  --stop-after-epochs 1
