#!/usr/bin/env python3
"""Check targets, schema membership, split-family isolation, and variant provenance."""

import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def options(question):
    if question["type"] == "noul":
        return {"true", "false"}
    if question["type"] == "choice":
        return {str(value) for value in question["criteria"]}
    if question["type"] == "score":
        return {str(index) for index in range(len(question["criteria"]))}
    raise ValueError(f"unsupported question type: {question['type']}")


def target_id(question):
    target = question["target"]
    if question["type"] == "noul":
        return "true" if target else "false"
    return str(target)


def main():
    splits = {name: read(DATA / name / "nimble.jsonl") for name in ("train", "calibration", "dev")}
    splits["test"] = read(DATA / "test/nimble_frozen.jsonl")
    family_sets = {name: {row["family"] for row in rows} for name, rows in splits.items()}
    for left, a in family_sets.items():
        for right, b in family_sets.items():
            if left < right:
                overlap = a & b
                if overlap:
                    raise AssertionError(f"family leakage between {left}/{right}: {len(overlap)}")
    question_count = record_count = 0
    for name, rows in splits.items():
        for row in rows:
            record_count += 1
            if row.get("source_id"):
                raise AssertionError(f"base split row unexpectedly marked as variant: {row['id']}")
            for question in row["questions"].values():
                question_count += 1
                if target_id(question) not in options(question):
                    raise AssertionError(f"target missing from schema in {row['id']}")
    variants = read(DATA / "augmented/train_schema_variants.jsonl")
    train_ids = {row["id"] for row in splits["train"]}
    if any(row.get("source_id") not in train_ids for row in variants):
        raise AssertionError("variant found without a source row in train split")
    source_orders = {row["id"]: list(row["questions"]) for row in splits["train"]}
    order_changed = sum(list(row["questions"]) != source_orders[row["source_id"]] for row in variants)
    if order_changed == 0:
        raise AssertionError("question-order variants did not change model input order")
    report = {"records": record_count, "questions": question_count, "train_only_variants": len(variants), "question_order_changed": order_changed, "family_disjoint": True, "targets_in_schema": True}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
