#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d weights/nimble-4b ]]; then echo "Checkpoint missing: train first." >&2; exit 2; fi
export HF_HOME="$(cd ../.. && pwd)/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
python scripts/eval_nimble.py
