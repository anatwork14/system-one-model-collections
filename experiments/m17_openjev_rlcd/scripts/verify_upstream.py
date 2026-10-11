#!/usr/bin/env python3
"""Pin and check the source files required by the M17 adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
EXPECTED_COMMIT = "22db40ad3333f5cf0bc5139097da70e0f8c05018"
MANIFEST = ROOT / "manifests/upstream_files.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def required_files():
    paths = [UPSTREAM / "README.md", UPSTREAM / "requirements.txt", UPSTREAM / "LICENSE",
             UPSTREAM / "data/gsm8k_verify.json", UPSTREAM / "scripts/common.sh",
             UPSTREAM / "scripts/two_stage.sh", UPSTREAM / "scripts/smoke_test.sh"]
    paths += sorted((UPSTREAM / "openjev_rlcd").rglob("*.py"))
    paths += sorted((UPSTREAM / "tests").rglob("*.py"))
    return sorted({p for p in paths if p.is_file()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    current = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
    if current.returncode == 0 and current.stdout.strip() != EXPECTED_COMMIT:
        raise SystemExit(f"upstream commit mismatch: {current.stdout.strip()}")
    manifest = {"repository": "https://github.com/ZimmyGao/openjev-rlcd",
                "commit": EXPECTED_COMMIT,
                "files": {p.relative_to(UPSTREAM).as_posix(): digest(p) for p in required_files()}}
    if args.write:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        expected = json.loads(MANIFEST.read_text())
        if expected != manifest:
            raise SystemExit("required source files differ from the pinned manifest")
    print(json.dumps({"status": "pass", "commit": EXPECTED_COMMIT,
                      "hashed_source_files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
