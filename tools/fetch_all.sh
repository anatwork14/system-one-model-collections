#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Create the project environment first: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r environment/base_requirements.txt" >&2
  exit 2
fi
FETCH_CONFIRM=yes .venv/bin/python tools/fetch_all.py
