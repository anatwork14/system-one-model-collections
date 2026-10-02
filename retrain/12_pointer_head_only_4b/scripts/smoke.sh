#!/usr/bin/env bash
set -euo pipefail
echo "Smoke inference is prepared by METHOD.md but requires downloaded artifacts and a supported runtime." >&2
echo "Method: $(basename "$(dirname "$0")")" >&2
exit 2
