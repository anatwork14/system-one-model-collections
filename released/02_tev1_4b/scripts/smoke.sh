#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
python "$HERE/scripts/infer.py" --model "$HERE/weights/model" \
  --input "$ROOT/evaluation/datasets/released_smoke.jsonl" \
  --output "$HERE/results/smoke.jsonl" "${@}"
