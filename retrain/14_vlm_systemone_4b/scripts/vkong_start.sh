#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$METHOD_DIR/../../.." && pwd)"
cd "$ROOT"
OUTPUT="$METHOD_DIR/results/v1-run"
export_artifacts() {
  local exit_code=$?
  trap - EXIT
  if [[ -f "$OUTPUT/last.pt" || -f "$OUTPUT/best.pt" ]]; then
    python3 "$METHOD_DIR/scripts/export_artifacts.py" --run-dir "$OUTPUT" || \
      echo "M14 artifact export failed; remote task disk is temporary." >&2
  fi
  exit "$exit_code"
}
trap export_artifacts EXIT
python3 "$METHOD_DIR/scripts/preflight.py"
python3 "$METHOD_DIR/scripts/vkong_smoke.py"
python3 "$METHOD_DIR/scripts/train.py" \
  --stage v1 --output "$OUTPUT" --epochs 5 --patience 2 \
  --learning-rate 0.0003 --weight-decay 0.01 --brier-weight 1.0 \
  --max-image-pixels 786432 --max-sequence-tokens 2048 --seed 17
python3 "$METHOD_DIR/scripts/predict.py" \
  --split model_validation --checkpoint "$OUTPUT/best.pt" \
  --output "$OUTPUT/eval/dev_predictions.jsonl"
