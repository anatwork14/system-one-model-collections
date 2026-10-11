#!/usr/bin/env python3
"""Validate original GSM8K-Verify rows and save split UID fingerprints."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
sys.path.insert(0, str(UPSTREAM))
from openjev_rlcd.tasks.gsm8k_verify import load_items  # noqa: E402

EXPECTED = "75d1cb9b69a5f2a1a84b9b003497e766217e73836e7808f088d6c0b3d32c467a"


def digest_ids(rows):
    raw = "\n".join(sorted(row["uid"] for row in rows)).encode()
    return hashlib.sha256(raw).hexdigest()


def question_key(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "manifests/dataset.json"))
    args = ap.parse_args()
    source = UPSTREAM / "data/gsm8k_verify.json"
    blob = source.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    if sha != EXPECTED:
        raise SystemExit(f"released dataset SHA mismatch: {sha}")
    raw = json.loads(blob)
    rows = raw["items"]
    required = {"uid", "split", "question", "gold", "proposed", "correct"}
    if not rows or any(not required.issubset(r) for r in rows):
        raise SystemExit("dataset is empty or row schema is incomplete")
    ids = [r["uid"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate dataset UIDs")
    if any(r["split"] not in {"train", "test"} for r in rows):
        raise SystemExit("unexpected source split label")
    for row in rows:
        if type(row["correct"]) is not bool:
            raise SystemExit(f"non-boolean label: {row['uid']}")
        try:
            gold = float(row["gold"].replace(",", ""))
            proposed = float(row["proposed"].replace(",", ""))
        except (TypeError, ValueError):
            raise SystemExit(f"invalid numeric answer: {row['uid']}")
        if row["correct"] != (abs(gold - proposed) < 1e-6):
            raise SystemExit(f"correctness label mismatch: {row['uid']}")
    source_train = [r for r in rows if r["split"] == "train"]
    source_test = [r for r in rows if r["split"] == "test"]
    train_questions = {question_key(r["question"]) for r in source_train}
    test_questions = {question_key(r["question"]) for r in source_test}
    if train_questions & test_questions:
        raise SystemExit(f"question overlap across original train/test: {len(train_questions & test_questions)}")
    splits = load_items(split_seed=0, n_train=4000, n_dev=500)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    pairs = [(set(r["uid"] for r in x), y) for x, y in ((train, "train"), (dev, "dev"), (test, "test"))]
    for i, (a, name_a) in enumerate(pairs):
        for b, name_b in pairs[i + 1:]:
            if a & b:
                raise SystemExit(f"overlap across RLCD {name_a}/{name_b} splits")
    if any(r["split"] != "test" for r in test) or any(r["split"] != "train" for r in train + dev):
        raise SystemExit("RLCD split assignment crossed source train/test boundary")
    report = {
        "source_file": "upstream/data/gsm8k_verify.json", "sha256": sha, "bytes": len(blob),
        "solver": raw.get("solver"), "source_seed": raw.get("seed"), "rows": len(rows),
        "source_split_counts": dict(Counter(r["split"] for r in rows)),
        "source_label_counts": {split: dict(Counter(str(r["correct"]).lower() for r in rows if r["split"] == split))
                                for split in ("train", "test")},
        "source_split_question_overlap": len(train_questions & test_questions),
        "rlcd_split_seed": 0,
        "rlcd_split_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "rlcd_split_uid_hashes": {"train": digest_ids(train), "dev": digest_ids(dev), "test": digest_ids(test)},
        "rlcd_uid_overlap": False, "test_source_split": "GSM8K original test",
        "selection_and_calibration_use_same_dev_pool": True,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
