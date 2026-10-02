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
    return validate(path)


def validate(path: Path) -> int:
    processed = path / "processed"
    jsonl_files = sorted(processed.glob("*.jsonl")) if processed.exists() else []
    if not jsonl_files:
        print(f"NOT VALIDATED: no normalized JSONL files found under {processed}")
        return 2
    errors = 0
    for file_path in jsonl_files:
        count = 0
        with file_path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"{file_path}:{line_number}: invalid JSON: {exc}")
                    errors += 1
                    continue
                absent = SCHEMA_KEYS - row.keys()
                if absent:
                    print(f"{file_path}:{line_number}: missing schema keys {sorted(absent)}")
                    errors += 1
                if isinstance(row, dict) and row.get("metadata", {}).get("dataset") is None:
                    print(f"{file_path}:{line_number}: metadata.dataset is required")
                    errors += 1
                if isinstance(row, dict):
                    question = row.get("question", {})
                    if question.get("type") == "choice":
                        criteria = question.get("criteria", {})
                        if str(row.get("target")) not in criteria:
                            print(f"{file_path}:{line_number}: choice target is not a criteria key")
                            errors += 1
                    image = row.get("state", {}).get("image")
                    if image:
                        ref, marker, location = image.partition("#")
                        target = (path / ref).resolve()
                        valid_location = (location.startswith("row=") and location[4:].isdigit()) or (
                            location.startswith("member=") and bool(location[7:]))
                        if not marker or not valid_location or not target.is_file():
                            print(f"{file_path}:{line_number}: invalid image reference {image!r}")
                            errors += 1
                count += 1
        print(f"{'FAIL' if errors else 'OK'} {file_path.relative_to(ROOT)} rows={count}")
    status_path = path / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["prepared"] = not errors
    status["validated"] = not errors
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
