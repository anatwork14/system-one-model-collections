#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$METHOD_DIR/../../.." && pwd)"
cd "$METHOD_DIR"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="python3"; fi
if [[ "${CONFIRM_TRAIN:-}" != "yes" ]]; then
  echo "This launches M14 V1 training. Inspect the data manifest and configs/variants.yaml first; set CONFIRM_TRAIN=yes to continue." >&2
  exit 2
fi
"$PYTHON" "$METHOD_DIR/scripts/preflight.py"
args=(--stage v1
  --output "${M14_OUTPUT:-$METHOD_DIR/results/v1-run}"
  --epochs "${M14_EPOCHS:-5}"
  --patience "${M14_PATIENCE:-2}"
  --max-image-pixels "${M14_MAX_IMAGE_PIXELS:-786432}")
if [[ "${M14_RESUME:-}" == "yes" ]]; then args+=(--resume); fi
exec "$PYTHON" "$METHOD_DIR/scripts/train.py" \
  "${args[@]}"
