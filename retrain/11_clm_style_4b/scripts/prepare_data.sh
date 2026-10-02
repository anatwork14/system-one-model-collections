#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_pairs.py --input ../10_nimble_4b/data/raw/train.jsonl --output data/processed/train_pairs.jsonl
python scripts/prepare_pairs.py --input ../10_nimble_4b/data/raw/eval.jsonl --output data/processed/eval_pairs.jsonl
