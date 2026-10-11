#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$METHOD_DIR/../../.." && pwd)"
cd "$METHOD_DIR"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="python3"; fi
OUTPUT="${M14_OUTPUT:-$METHOD_DIR/results/v1-run}"
mkdir -p "$OUTPUT/eval"
exec "$PYTHON" "$METHOD_DIR/scripts/predict.py" \
  --split model_validation \
  --checkpoint "$OUTPUT/best.pt" \
  --output "$OUTPUT/eval/dev_predictions.jsonl"
