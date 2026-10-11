#!/usr/bin/env python3
"""Create the AdaS1 controlled P0 data from M10's train/dev split only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHOD = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "retrain/10_nimble_4b/data/inner_split"
EXPECTED = {
    "train": (2268, "96120cc63f6c149cb4bca4c0066a8f3c2ec124a3aa1cf13f82e514393c35304d"),
    "dev": (408, "96f86f36bfe7512fdc8dcadc8c313e9d83cb2cf5625224d4633e2bf7f17a6a59"),
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonicalize(row: dict) -> dict:
    source = row["input"]
    questions = {}
    for question_id, question in source["questions"].items():
        questions[str(question_id)] = {**question, "target": row["reference"]["target"]}
    return {
        "id": str(row["id"]),
        "family": str(row.get("source_family", row.get("family", row["id"]))),
        "source_id": str(row.get("provenance", {}).get("source_id", row["id"])),
        "state": source["state"],
        "questions": questions,
        "source": "nimble_controlled_train",
        "label_status": row.get("quality_status", "source_labeled"),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=METHOD / "data/controlled")
    args = parser.parse_args()
    train_path, dev_path = SOURCE / "train.jsonl", SOURCE / "validation.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("Prepare the corrected M10 inner split before M16: run retrain/10_nimble_4b/scripts/prepare.sh")
    for name, path in (("train", train_path), ("dev", dev_path)):
        expected_count, expected_hash = EXPECTED[name]
        actual_count, actual_hash = len(read_rows(path)), sha256(path)
        if actual_count != expected_count or actual_hash != expected_hash:
            raise ValueError(f"Pinned M10 {name} split changed: rows={actual_count}, sha256={actual_hash}")
    splits = {"train": [canonicalize(row) for row in read_rows(train_path)],
              "dev": [canonicalize(row) for row in read_rows(dev_path)]}
    ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    families = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    source_ids = {name: {row["source_id"] for row in rows} for name, rows in splits.items()}
    if not all(splits.values()) or ids["train"] & ids["dev"] or families["train"] & families["dev"] or source_ids["train"] & source_ids["dev"]:
        raise ValueError("P0 train/dev must be nonempty and disjoint by record, family, and source id")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "dataset": "nimble-m16-p0-controlled-v1",
        "source": "retrain/10_nimble_4b/data/inner_split/{train,validation}.jsonl",
        "source_sha256": {"train": sha256(train_path), "dev": sha256(dev_path)},
        "controlled_sha256": {
            name: sha256(args.output / f"{name}.jsonl") for name in splits
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "family_disjoint": True,
        "source_id_disjoint": True,
        "outer_324_eval_read": False,
        "label_caveat": "Synthetic/model-checked Nimble labels; not fully human reviewed.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
