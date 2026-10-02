#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -s data/raw/train.jsonl || { echo "Run tools/fetch_all.sh first; Nimble train data is absent." >&2; exit 2; }
test -s data/raw/eval.jsonl || { echo "Nimble frozen evaluation data is absent." >&2; exit 2; }
test -s data/manifests/manifest.json || { echo "Data manifest is absent." >&2; exit 2; }
PYTHONPATH="${PWD}/upstream${PYTHONPATH:+:${PYTHONPATH}}" python -m nimble.training.verify_dataset
python - <<'PY'
import json
from pathlib import Path
for filename, expected in (("train.jsonl", 2676), ("eval.jsonl", 324)):
    n = sum(1 for line in Path("data/raw", filename).open(encoding="utf-8") if line.strip())
    if n != expected:
        raise SystemExit(f"{filename}: expected {expected} rows, got {n}")
print("Nimble data counts and upstream verification passed.")
PY
python scripts/to_kev.py --input data/raw/train.jsonl --output data/processed/kev_train.jsonl
python scripts/to_kev.py --input data/raw/eval.jsonl --output data/processed/kev_eval.jsonl
