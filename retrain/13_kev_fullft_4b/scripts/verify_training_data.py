#!/usr/bin/env python3
"""Verify the prebuilt full-FT input against the shared inner-split manifest."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/inner_split.json"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(path, manifest):
    rows = read_rows(path)
    expected = manifest["files"].get(path.name)
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows or not expected or expected["rows"] != len(rows) or expected["sha256"] != actual_hash:
        raise ValueError(f"Training artifact does not match split manifest: {path}")
    return rows


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = verify(ROOT / "data/processed/kev_train_inner.jsonl", manifest)
    dev = verify(ROOT / "data/processed/kev_dev_inner.jsonl", manifest)
    ids = lambda rows: {row["_meta"]["id"] for row in rows}
    families = lambda rows: {row["_meta"]["source_family"] for row in rows}
    if ids(train) & ids(dev) or families(train) & families(dev):
        raise ValueError("Full-FT train/dev inputs overlap by id or source family")
    if any(not row["_meta"].get("source_family") for row in train + dev):
        raise ValueError("Full-FT inputs require source-family metadata")
    print(json.dumps({"train_records": len(train), "dev_records": len(dev),
                      "train_dev_ids_disjoint": True, "train_dev_families_disjoint": True,
                      "outer_eval_included": False}, indent=2))


if __name__ == "__main__":
    main()
