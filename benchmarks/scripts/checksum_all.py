#!/usr/bin/env python3
"""Write SHA-256 lists for local canonical raw-data files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    for key, item in registry.items():
        if not item["raw_unique"]:
            continue
        path = ROOT / item["path"]
        raw = path / "raw"
        files = sorted(file for file in raw.rglob("*") if file.is_file()) if raw.exists() else []
        lines = [f"{digest(file)}  {file.relative_to(path).as_posix()}" for file in files]
        (path / "checksums.sha256").write_text("\n".join(lines) + ("\n" if lines else "# No raw files downloaded.\n"))
        if files:
            (ROOT / "checksums" / f"{key}.sha256").write_text("\n".join(lines) + "\n")
        print(f"{key}: {len(files)} raw file(s)")


if __name__ == "__main__":
    main()
