#!/usr/bin/env python3
"""Pack pilot images into VKong-safe <50 MiB source chunks."""
import hashlib
import json
import tarfile
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
IMAGE_DIR = METHOD / "data/images/flowers102"
PART_DIR = METHOD / "data/vkong_image_parts"
PART_BYTES = 45 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    relative_images = set()
    for split in ("train", "model_validation", "calibration", "test"):
        path = METHOD / "data/processed" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                relative_images.update(json.loads(line)["images"])
    images = [METHOD / "data" / relative for relative in sorted(relative_images)]
    if len(images) != 8171:
        raise SystemExit(f"Expected 8,171 audited images; found {len(images)}")
    PART_DIR.mkdir(parents=True, exist_ok=True)
    for old in PART_DIR.glob("flowers-*.part"):
        old.unlink()
    archive_path = PART_DIR / ".flowers.tar.tmp"
    with tarfile.open(archive_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for image in images:
            archive.add(image, arcname=f"images/flowers102/{image.name}", recursive=False)
    archive_hash = sha256(archive_path)
    parts = []
    with archive_path.open("rb") as source:
        index = 0
        while True:
            payload = source.read(PART_BYTES)
            if not payload:
                break
            name = f"flowers-{index:02d}.part"
            part_path = PART_DIR / name
            part_path.write_bytes(payload)
            parts.append({"path": name, "bytes": len(payload), "sha256": sha256(part_path)})
            index += 1
    archive_bytes = archive_path.stat().st_size
    archive_path.unlink()
    manifest = {"format": 1, "image_count": len(images), "archive_bytes": archive_bytes,
                "archive_sha256": archive_hash, "part_bytes_limit": PART_BYTES, "parts": parts}
    (PART_DIR / "parts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image_count": len(images), "archive_bytes": archive_bytes,
                      "parts": len(parts), "largest_part_bytes": max(part["bytes"] for part in parts)}, indent=2))


if __name__ == "__main__":
    main()
