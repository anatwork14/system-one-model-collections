#!/usr/bin/env bash
set -euo pipefail
method_dir="$(cd "$(dirname "$0")/.." && pwd)"
python "$method_dir/src/train.py" --data "$method_dir/data/train/nimble.jsonl" --stage "${CLEF_STAGE:-head_only}" --rank 256 --epochs "${CLEF_EPOCHS:-1}" --output "$method_dir/checkpoints/${CLEF_STAGE:-head_only}"
