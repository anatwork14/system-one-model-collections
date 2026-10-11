#!/usr/bin/env python3
"""Shared per-dataset command guard and normalized-data validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_KEYS = {"id", "state", "question", "target", "metadata"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--action", required=True, choices=["download", "prepare", "validate"])
    parser.add_argument("--read-only", action="store_true",
                        help="validate prepared JSONL without refreshing status.json")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    if args.dataset not in registry:
        parser.error(f"unknown dataset: {args.dataset}")
    item = registry[args.dataset]
    path = ROOT / item["path"]
    if args.action == "download":
        downloader = ROOT / "scripts" / "download_core.py"
        return subprocess.run([sys.executable, str(downloader), "--dataset", args.dataset], check=False).returncode
    if args.action == "prepare":
        preparer = ROOT / "scripts" / "prepare_core.py"
        return subprocess.run([sys.executable, str(preparer), "--dataset", args.dataset], check=False).returncode
    return validate(path, update_status=not args.read_only)


def validate(path: Path, update_status: bool = True) -> int:
    processed = path / "processed"
    jsonl_files = sorted(processed.glob("*.jsonl")) if processed.exists() else []
    if not jsonl_files:
        print(f"NOT VALIDATED: no normalized JSONL files found under {processed}")
        return 2
    errors = 0
    for file_path in jsonl_files:
        count = 0
        file_errors = 0
        seen_ids = set()
        with file_path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{file_path}:{line_number}: invalid JSON: {exc}")
                    errors += 1
                    file_errors += 1
                    continue
                if not isinstance(row, dict):
                    print(f"{file_path}:{line_number}: record must be an object")
                    errors += 1
                    file_errors += 1
                    continue
                absent = SCHEMA_KEYS - row.keys()
                if absent:
                    print(f"{file_path}:{line_number}: missing schema keys {sorted(absent)}")
                    errors += 1
                    file_errors += 1
                if isinstance(row, dict) and row.get("metadata", {}).get("dataset") is None:
                    print(f"{file_path}:{line_number}: metadata.dataset is required")
                    errors += 1
                    file_errors += 1
                if isinstance(row, dict):
                    if row.get("id") in seen_ids:
                        print(f"{file_path}:{line_number}: duplicate id {row.get('id')!r}")
                        errors += 1
                        file_errors += 1
                    seen_ids.add(row.get("id"))
                    question = row.get("question", {})
                    if question.get("type") == "choice":
                        criteria = question.get("criteria", {})
                        if str(row.get("target")) not in criteria:
                            print(f"{file_path}:{line_number}: choice target is not a criteria key")
                            errors += 1
                            file_errors += 1
                    image = row.get("state", {}).get("image")
                    if image:
                        ref, marker, location = image.partition("#")
                        target = (path / ref).resolve()
                        valid_location = (location.startswith("row=") and location[4:].isdigit()) or (
                            location.startswith("member=") and bool(location[7:]))
                        if not marker or not valid_location or not target.is_file():
                            print(f"{file_path}:{line_number}: invalid image reference {image!r}")
                            errors += 1
                            file_errors += 1
                count += 1
        print(f"{'FAIL' if file_errors else 'OK'} {file_path.relative_to(ROOT)} rows={count}")
    if path.parent.name == "openworld":
        errors += validate_openworld(path)
    if update_status:
        status_path = path / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status["prepared"] = not errors
        status["validated"] = not errors
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 1 if errors else 0


def validate_openworld(path: Path) -> int:
    manifest_path = path / "splits/known75/split_manifest.json"
    if not manifest_path.is_file():
        print(f"FAIL {path.relative_to(ROOT)}: known75 split manifest missing")
        return 1
    manifest = json.loads(manifest_path.read_text())
    known = set(manifest.get("known_classes", []))
    unseen = set(manifest.get("unseen_classes", []))
    errors = 0
    all_classes = set(manifest.get("all_classes", []))
    if not known or not unseen or known & unseen or (all_classes and known | unseen != all_classes):
        print(f"FAIL {path.relative_to(ROOT)}: known/unseen class partition is empty or overlapping")
        return 1
    processed = path / "processed"
    train_sources = set()
    evaluation_sources = set()
    for filename, protocol in (("train_known.jsonl", "train"), ("closed.jsonl", "closed"),
                               ("unknown_absent.jsonl", "unknown_absent"),
                               ("unseen_supplied.jsonl", "unseen_supplied")):
        file_path = processed / filename
        if not file_path.is_file():
            print(f"FAIL {file_path.relative_to(ROOT)}: expected protocol split missing")
            errors += 1
            continue
        file_errors = 0
        with file_path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                criteria = row.get("question", {}).get("criteria", {})
                choices = set(criteria)
                target = str(row.get("target"))
                metadata = row.get("metadata", {})
                source_id = metadata.get("source_id")
                if source_id is not None:
                    (train_sources if protocol == "train" else evaluation_sources).add(str(source_id))
                row_error = None
                if protocol == "train":
                    if not choices or not choices <= known or target not in known or metadata.get("true_class", target) not in known:
                        row_error = "training rows must use known classes only"
                elif protocol == "closed":
                    if (choices != known or target not in known or metadata.get("protocol") != "OW-0-closed"):
                        row_error = "closed-set rows must use exactly the known classes"
                elif protocol == "unknown_absent":
                    true_class = metadata.get("true_class")
                    target_mapping_valid = (target in known if target != "UNKNOWN"
                                            else true_class in unseen or metadata.get("native_clinc_oos") is True)
                    if (choices != known | {"UNKNOWN"} or not target_mapping_valid
                            or (true_class is not None and true_class not in known | unseen)
                            or metadata.get("protocol") != "OW-1-unknown-absent"):
                        row_error = "unknown-absent rows must use known classes plus UNKNOWN"
                else:
                    supplied = row.get("metadata", {}).get("supplied_class")
                    expected = known | {"UNKNOWN", supplied} if supplied else set()
                    if (supplied not in unseen or choices != expected or target != supplied
                            or metadata.get("protocol") != "OW-2-runtime-supplied"):
                        row_error = "runtime-supplied rows need exactly known, supplied unseen, and UNKNOWN options; target must be supplied class"
                if row_error:
                    print(f"{file_path}:{line_no}: {row_error}")
                    errors += 1
                    file_errors += 1
                    break
        print(f"{'FAIL' if file_errors else 'OK'} {file_path.relative_to(ROOT)} protocol={protocol}")
    overlap = train_sources & evaluation_sources
    if overlap:
        print(f"FAIL {path.relative_to(ROOT)}: train/evaluation source IDs overlap ({len(overlap)})")
        errors += 1
    if errors:
        print(f"FAIL {path.relative_to(ROOT)} open-world protocol invariants: {errors} error(s)")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
