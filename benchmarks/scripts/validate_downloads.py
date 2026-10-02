#!/usr/bin/env python3
"""Validate downloaded raw payload counts, protocol artifacts, and checksums."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parquet_rows(directory: Path, split: str) -> int:
    files = sorted((directory / "data").glob(f"{split}-*.parquet"))
    if not files:
        files = sorted((directory / "plain_text").glob(f"{split}-*.parquet"))
    if not files:
        files = sorted(directory.glob(f"{split}-*.parquet"))
    if not files:
        raise ValueError(f"missing parquet split {split} under {directory}")
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def validate(key: str, item: dict) -> tuple[dict, list[str]]:
    path = ROOT / item["path"]
    raw = path / "raw"
    details: dict = {}
    errors: list[str] = []
    if key == "jevbench_public":
        splits = {name: sum(1 for line in (raw / "public" / f"{name}.jsonl").open() if line.strip())
                  for name in ("original", "easy", "hard")}
        details["splits"] = splits
        if splits != {"original": 72, "easy": 48, "hard": 111}:
            errors.append(f"JevBench public counts differ from 72/48/111: {splits}")
    elif key == "banking77":
        counts, labels = {}, set()
        for split in ("train", "test"):
            with (raw / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            counts[split] = len(rows)
            labels.update(row.get("category") or row.get("label") for row in rows)
        details.update({"splits": counts, "classes": len(labels)})
        if counts != {"train": 10003, "test": 3080} or len(labels) != 77:
            errors.append(f"BANKING77 expected 10003/3080 rows and 77 labels, got {counts}, {len(labels)}")
    elif key == "clinc150_oos":
        source = json.loads((raw / "data_full.json").read_text())
        counts = {split: len(rows) for split, rows in source.items()}
        details["source_splits"] = counts
        expected = {"train": 15000, "val": 3000, "test": 4500, "oos_train": 100, "oos_val": 100, "oos_test": 1000}
        if counts != expected:
            errors.append(f"CLINC split counts differ from expected: {counts}")
    elif key == "boolq":
        counts = {name: parquet_rows(raw, name) for name in ("train", "validation")}
        details["splits"] = counts
        if counts != {"train": 9427, "validation": 3270}:
            errors.append(f"BoolQ split counts differ from 9427/3270: {counts}")
    elif key == "goemotions":
        counts = {name: parquet_rows(raw / "simplified", name) for name in ("train", "validation", "test")}
        source_rows = []
        for part in ("1", "2", "3"):
            with (raw / f"goemotions_{part}.csv").open(encoding="utf-8", newline="") as stream:
                source_rows.extend(csv.DictReader(stream))
        annotation_count = len(source_rows)
        raw_count = len({row["id"] for row in source_rows})
        annotation_rows = parquet_rows(raw / "raw", "train")
        details.update({"raw_unique_examples": raw_count, "planning_note_examples": 58009,
                        "annotation_rows": annotation_count,
                        "hub_annotation_rows": annotation_rows,
                        "filtered_splits": counts})
        # The plan's 58,009 figure is two below the distinct-ID count in the
        # pinned Google raw CSV files. Preserve the observed source count and
        # separately validate the canonical annotation-row and filtered-split
        # counts; do not silently rewrite the source grain to fit the plan.
        if raw_count != 58011:
            errors.append(f"GoEmotions pinned raw CSV unique-ID count differs from 58011: {raw_count}")
        if annotation_count != 211225 or annotation_rows != annotation_count:
            errors.append(f"GoEmotions annotation count differs from 211225: csv={annotation_count}, hub={annotation_rows}")
        if counts != {"train": 43410, "validation": 5426, "test": 5427}:
            errors.append(f"GoEmotions filtered counts differ from expected: {counts}")
    elif key == "when2call":
        count = sum(1 for line in (raw / "when2call_test_mcq.jsonl").open() if line.strip())
        details["test_requests"] = count
        if count != 3652:
            errors.append(f"When2Call expected 3652 requests, got {count}")
    elif key == "bfcl":
        selected = {}
        for filename in ("BFCL_v4_simple_python.json", "BFCL_v4_irrelevance.json"):
            records = [json.loads(line) for line in (raw / filename).read_text().splitlines() if line.strip()]
            selected[filename] = len(records)
            if not records:
                errors.append(f"BFCL file is empty: {filename}")
        details["selected_records"] = selected
    elif key in {"banking77_bolt", "clinc150_bolt", "hwu64_bolt", "stackoverflow_bolt"}:
        folder = ROOT / item["path"] / "splits" / "known75"
        label_file = next(folder.glob("*_label.list"), None)
        known_file = next(folder.glob("*_label_known_0.75.list"), None)
        if not label_file or not known_file:
            errors.append("missing BOLT class list or known75 split list")
        else:
            all_labels = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
            known = [line.strip() for line in known_file.read_text().splitlines() if line.strip()]
            unseen = sorted(set(all_labels) - set(known))
            details.update({"all_classes": len(all_labels), "known_classes": len(known), "unseen_classes": len(unseen)})
            if len(set(all_labels)) != len(all_labels) or len(set(known)) != len(known):
                errors.append("duplicate class names in BOLT lists")
            if not set(known).issubset(set(all_labels)) or set(known) & set(unseen):
                errors.append("BOLT known/unseen class sets are invalid")
            expected_classes = int(item["classes"])
            if len(all_labels) != expected_classes:
                errors.append(f"expected {expected_classes} labels, got {len(all_labels)}")
        if key in {"hwu64_bolt", "stackoverflow_bolt"}:
            origin = raw / "origin_data"
            counts = {split: sum(1 for line in (origin / f"{split}.tsv").open(encoding="utf-8") if line.strip())
                      for split in ("train", "dev", "test")}
            counts = {split: count - 1 for split, count in counts.items()}  # BOLT TSVs have one header row.
            details["raw_splits"] = counts
            if sum(counts.values()) != item["expected_examples"]:
                errors.append(f"BOLT raw split total {sum(counts.values())} != {item['expected_examples']}")
    elif key == "cifar10":
        counts = {name: parquet_rows(raw, name) for name in ("train", "test")}
        details["splits"] = counts
        if counts != {"train": 50000, "test": 10000}:
            errors.append(f"CIFAR-10 split counts differ from expected: {counts}")
    elif key == "oxford_iiit_pets":
        counts = {name: parquet_rows(raw, name) for name in ("train", "test")}
        details["splits"] = counts
        if counts != {"train": 3680, "test": 3669}:
            errors.append(f"Oxford Pets counts differ from official trainval/test: {counts}")
    elif key == "flowers102":
        counts = {name: parquet_rows(raw, name) for name in ("train", "validation", "test")}
        details["splits"] = counts
        if counts != {"train": 1020, "validation": 1020, "test": 6149}:
            errors.append(f"Flowers-102 official split counts differ from expected: {counts}")

    lock = yaml.safe_load((path / "SOURCE.lock.yaml").read_text())
    expected_hashes = lock.get("raw_hashes", {})
    actual_files = {p.relative_to(path).as_posix(): digest(p) for p in raw.rglob("*") if p.is_file()}
    missing_hashes = sorted(set(actual_files) - set(expected_hashes))
    mismatched = sorted(name for name, value in expected_hashes.items() if actual_files.get(name) != value)
    details.update({"raw_files": len(actual_files), "sha256_verified": not missing_hashes and not mismatched,
                    "unhashed_files": missing_hashes, "checksum_mismatches": mismatched})
    if missing_hashes:
        errors.append(f"raw files without SHA-256 entries: {missing_hashes[:5]}")
    if mismatched:
        errors.append(f"raw checksum mismatch: {mismatched[:5]}")
    return details, errors


def main() -> int:
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    report = {"status": "passed", "datasets": {}}
    for key, item in registry.items():
        try:
            details, errors = validate(key, item)
        except Exception as exc:
            details, errors = {}, [f"validation could not inspect payload: {exc}"]
        report["datasets"][key] = {"validated": not errors, "details": details, "errors": errors}
        print(f"{'PASS' if not errors else 'FAIL'} {key}: {details}")
        status_path = ROOT / item["path"] / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status["validated"] = not errors
        status["examples"] = item.get("expected_examples") if not errors else None
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if errors:
            report["status"] = "failed"
    destination = ROOT / "manifests" / "download_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {destination.relative_to(ROOT)}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
