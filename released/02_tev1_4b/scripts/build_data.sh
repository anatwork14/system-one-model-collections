#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STAGE="$HERE/data/build_src"
OUT="$HERE/data/original_recipe/new-v1"
if [[ -e "$STAGE" ]]; then
  [[ -f "$STAGE/pyproject.toml" && -f "$STAGE/build_all.py" && -f "$STAGE/uv.lock" ]] || {
    echo "Existing Tev1 staging directory is incomplete or unrecognized: $STAGE" >&2; exit 2;
  }
else
  mkdir -p "$STAGE"
  git -C "$HERE/upstream" archive HEAD | tar -x -C "$STAGE"
fi
cd "$STAGE"
uv sync --locked
uv run python fetch_sources.py
if [[ ! -f data/new-v1/manifest.json ]]; then
  uv run python build_all.py
fi
uv run python validate_dataset.py data/v1
if [[ -e "$OUT" ]]; then
  diff -qr data/new-v1 "$OUT" >/dev/null || { echo "Existing copied Tev1 data differs from pinned build output: $OUT" >&2; exit 2; }
else
  mkdir -p "$(dirname "$OUT")"
  cp -a data/new-v1 "$OUT"
fi
diff -qr data/new-v1 "$OUT" >/dev/null
printf 'Tev1 dataset copied to %s\n' "$OUT"
printf 'Source commit: '
git -C "$HERE/upstream" rev-parse HEAD
