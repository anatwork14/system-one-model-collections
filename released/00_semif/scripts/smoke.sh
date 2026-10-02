#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/semif-score" ]]; then echo "Install first with scripts/install.sh" >&2; exit 2; fi
"$ROOT/.venv/bin/semif-score" --mode direct \
  --model "$ROOT/shared/models/qwen3.5-4b" \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input "$HERE/upstream/examples/decisions.jsonl" \
  --output "$HERE/results/smoke.jsonl"
