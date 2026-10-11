#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 ../10_nimble_4b/scripts/prepare_inner_split.py \
  --source-dir ../10_nimble_4b/data/raw \
  --output-dir ../10_nimble_4b/data/inner_split \
  --seed 17
python3 -c 'import json, pathlib; files=[pathlib.Path("data/processed/train_pairs_inner.jsonl"), pathlib.Path("data/processed/dev_pairs_inner.jsonl")]; rows=[[json.loads(line) for line in p.read_text().splitlines() if line.strip()] for p in files]; train_ids={r["id"] for r in rows[0]}; dev_ids={r["id"] for r in rows[1]}; train_families={r["family"] for r in rows[0]}; dev_families={r["family"] for r in rows[1]}; assert train_ids and dev_ids and not train_ids & dev_ids and not train_families & dev_families; print({str(p): len(rs) for p, rs in zip(files, rows)})'
