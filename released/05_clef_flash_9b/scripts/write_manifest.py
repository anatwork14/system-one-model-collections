#!/usr/bin/env python3
"""Write a SHA-256 manifest for the pinned Clef-Flash local files."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "weights" / "architecture", ROOT / "weights" / "model"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = []
    seen = set()
    for directory in FILES:
        if not directory.exists():
            raise SystemExit(f"required model directory missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.endswith((".incomplete", ".pyc")) or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in seen or ".cache" in path.parts:
                continue
            seen.add(relative)
            manifest.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    destination = ROOT / "weights" / "MANIFEST.json"
    payload = {
        "repo": "Cloudflare/clef-flash",
        "revision": "17f0b0ad64efb65d273590632833508766b2aae6",
        "files": manifest,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote SHA-256 manifest for {len(manifest)} files to {destination}")


if __name__ == "__main__":
    main()
