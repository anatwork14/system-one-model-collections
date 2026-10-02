#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
uv pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$(dirname "$0")/../upstream"
