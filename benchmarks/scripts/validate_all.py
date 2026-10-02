#!/usr/bin/env python3
"""Check benchmark metadata, raw references, checksums, and processed records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    failures: list[str] = []
    report: dict[str, dict] = {}
    raw_owners: dict[str, str] = {}
    for key, item in registry.items():
        path = ROOT / item["path"]
        required = [path / "DATASET.md", path / "SOURCE.lock.yaml", path / "checksums.sha256"]
        missing = [str(file.relative_to(ROOT)) for file in required if not file.is_file()]
        lock = yaml.safe_load((path / "SOURCE.lock.yaml").read_text()) if (path / "SOURCE.lock.yaml").exists() else {}
        raw_path = path / "raw"
        if item["raw_unique"]:
            raw_path.mkdir(exist_ok=True)
            raw_owners[str(raw_path.resolve())] = key
            downloaded = any(file.is_file() for file in raw_path.rglob("*"))
        else:
            ref_file = path / "raw_ref.yaml"
            if not ref_file.is_file():
                missing.append(str(ref_file.relative_to(ROOT)))
                downloaded = False
            else:
                ref = yaml.safe_load(ref_file.read_text())
                ref_path = (path / ref["raw_path"]).resolve()
                downloaded = ref_path.is_dir() and any(file.is_file() for file in ref_path.rglob("*"))
                if not ref_path.is_relative_to(ROOT.resolve()):
                    failures.append(f"{key}: raw_ref escapes benchmarks directory")
        hashes = parse_checksums(path / "checksums.sha256")
        actual = {file.relative_to(path).as_posix(): sha256(file)
                  for file in raw_path.rglob("*") if file.is_file()} if raw_path.exists() else {}
        mismatches = [name for name, value in hashes.items() if actual.get(name) != value]
        absent_hashes = sorted(set(actual) - hashes.keys())
        status = "validated" if downloaded and not missing and not mismatches and not absent_hashes else "planned"
        report[key] = {"path": item["path"], "downloaded": downloaded, "validated": status == "validated",
                       "source_revision": lock.get("source", {}).get("revision"), "missing_metadata": missing,
                       "checksum_mismatches": mismatches, "files_missing_checksums": absent_hashes}
        if missing or mismatches or absent_hashes:
            failures.append(f"{key}: metadata/checksum issue")
        print(f"{status.upper():10} {key}: downloaded={downloaded}, metadata_missing={len(missing)}, hash_mismatch={len(mismatches)}, unhashed={len(absent_hashes)}")
    output = {"status": "failed" if failures else "passed", "datasets": report, "failures": failures}
    if args.write_report:
        target = ROOT / "manifests" / "validation_report.json"
        target.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote {target.relative_to(ROOT)}")
    return 1 if failures else 0


def parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        result[name.lstrip("* ")] = digest
    return result


if __name__ == "__main__":
    raise SystemExit(main())
