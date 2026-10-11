#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$WORKSPACE_ROOT"
OUTPUT="${M16_OUTPUT:-/data/adas1-p0/$(date -u +%Y%m%d-%H%M%S)}"
LAYERS="4,8,12,16,20,24,28,32"
while (($#)); do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname -- "$OUTPUT")"
export PYTHONUNBUFFERED=1
exec python3 -u "$SCRIPT_DIR/00_probe_all_layers.py" \
  --base-model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output "$OUTPUT" \
  --layers "$LAYERS"
