#!/usr/bin/env python3
"""Download the pinned public model into the temporary VKong task disk."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import torch
import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download, snapshot_download

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
MODEL = ROOT / "shared/models/qwen3.5-4b"
DATA = METHOD / "data"
REPO = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FLOWERS_MIRROR_REPO = "Donghyun99/Oxford-Flower-102"
FLOWERS_MIRROR_REVISION = "7e43fe37eeb9278138ab0bd2d0abb10d0f703d09"
FLOWERS_PARQUETS = (
    "data/train-00000-of-00001-95813d8dab877ff5.parquet",
    "data/validation-00000-of-00001-a2ccf8138610462c.parquet",
    "data/test-00000-of-00001-1cc6f065cd5391ec.parquet",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_images() -> int:
    audit_path = DATA / "manifests/leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise SystemExit("M14 image extraction requires the passed, pinned leakage audit")
    expected = audit.get("image_files", {})
    expected_by_sha = {item["sha256"]: relative for relative, item in expected.items()}
    if len(expected_by_sha) != len(expected):
        raise SystemExit("M14 leakage audit contains duplicate image hashes")

    print("Downloading Flowers-102 files from the pinned Hub mirror revision", flush=True)
    parquet_paths = [
        hf_hub_download(
            repo_id=FLOWERS_MIRROR_REPO,
            filename=filename,
            repo_type="dataset",
            revision=FLOWERS_MIRROR_REVISION,
        )
        for filename in FLOWERS_PARQUETS
    ]
    image_root = DATA / "images/flowers102"
    image_root.mkdir(parents=True, exist_ok=True)
    recovered: set[str] = set()
    for parquet_path in parquet_paths:
        parquet_file = parquet.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=128, columns=["image"]):
            for item in batch.column(0).to_pylist():
                payload = item.get("bytes") if isinstance(item, dict) else None
                if not payload:
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                relative = expected_by_sha.get(digest)
                if relative is None or relative in recovered:
                    continue
                safe_path = PurePosixPath(relative)
                if len(safe_path.parts) != 3 or safe_path.parts[:2] != ("images", "flowers102"):
                    raise SystemExit(f"Unexpected audited image path: {relative}")
                target = image_root / safe_path.name
                temporary = target.with_suffix(".jpg.tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, target)
                recovered.add(relative)
    if recovered != set(expected):
        missing = len(set(expected) - recovered)
        raise SystemExit(f"Pinned Flowers mirror did not match the audited corpus: {missing} image hashes missing")
    print(f"Recovered {len(recovered)} audited images by SHA-256 from the pinned mirror", flush=True)
    return len(recovered)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Selected VKong machine has no CUDA PyTorch device")
    gpu = torch.cuda.get_device_properties(0)
    vram_gib = gpu.total_memory / 1024**3
    if vram_gib < 23:
        raise SystemExit(f"Selected VKong GPU has {vram_gib:.1f} GiB runtime-visible VRAM; M14 V1 requires at least 23 GiB")
    print(f"CUDA preflight passed: {torch.cuda.get_device_name(0)}, {vram_gib:.1f} GiB VRAM", flush=True)
    reconstruct_images()
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(MODEL))
    marker = MODEL / ".workspace-revision"
    marker.write_text(f"{REPO}@{REVISION}\n", encoding="utf-8")
    print(f"Pinned model ready: {REPO}@{REVISION}", flush=True)


if __name__ == "__main__":
    main()
