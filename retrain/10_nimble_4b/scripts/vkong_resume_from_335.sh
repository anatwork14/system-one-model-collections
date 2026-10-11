#!/usr/bin/env bash
set -euo pipefail
echo "This legacy checkpoint used the full training split and the outer evaluation set. Do not resume it. Start a fresh run with scripts/train.sh after reviewing its clean inner split." >&2
exit 2
