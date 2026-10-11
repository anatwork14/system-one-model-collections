#!/usr/bin/env bash
set -euo pipefail
method_dir="$(cd "$(dirname "$0")/.." && pwd)"
stage="${CLEF_STAGE:-head_only}"
output="$method_dir/checkpoints/clef4b-${stage}-clean"
[[ ! -e "$output" ]] || { echo "Output already exists: $output. Pick a fresh run directory; checkpoints are never overwritten." >&2; exit 2; }
python3 "$method_dir/scripts/validate_data.py" --training-only
python3 "$method_dir/src/train.py" \
  --data "$method_dir/data/train/nimble.jsonl" \
  --augmented "$method_dir/data/augmented/train_schema_variants.jsonl" \
  --validation "$method_dir/data/dev/nimble.jsonl" \
  --stage "$stage" --rank "${CLEF_RANK:-256}" --epochs "${CLEF_EPOCHS:-20}" \
  --output "$output"
