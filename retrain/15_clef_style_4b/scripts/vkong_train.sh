#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${CLEF_STAGE:-head_only}"
OUTPUT="/data/checkpoints/clef4b-${STAGE}-clean"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT. Choose a new run directory; this launcher never deletes checkpoints." >&2; exit 2; }
python3 scripts/validate_data.py --training-only
export PYTHONUNBUFFERED=1
python3 -u src/train.py \
  --data data/train/nimble.jsonl \
  --augmented data/augmented/train_schema_variants.jsonl \
  --validation data/dev/nimble.jsonl \
  --base-model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output "$OUTPUT" \
  --stage "$STAGE" \
  --epochs "${CLEF_EPOCHS:-20}" \
  --rank "${CLEF_RANK:-256}" \
  --seed 17
