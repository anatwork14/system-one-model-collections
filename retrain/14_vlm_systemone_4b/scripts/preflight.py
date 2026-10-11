#!/usr/bin/env python3
"""Report whether M14's data, model, trainer, and hardware launch gates are met."""

import json
import subprocess
import sys
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
EXPECTED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> int:
    blockers: list[str] = []
    model_path = ROOT / "shared/models/qwen3.5-4b"
    revision_file = model_path / ".workspace-revision"
    if not model_path.is_dir():
        blockers.append(f"Pinned base model is missing: {model_path}")
    elif not revision_file.exists() or EXPECTED_REVISION not in revision_file.read_text():
        blockers.append("Local base model revision does not match the M14 pin")

    manifest_path = METHOD / "data/manifests/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ready":
        blockers.append("Image decision data manifest is not ready")
    checks = manifest.get("required_checks", {})
    for key in (
        "image_use_terms_verified",
        "image_provenance_recorded",
        "test_excluded_from_training_and_model_selection",
    ):
        if checks.get(key) is not True:
            blockers.append(f"Dataset gate is incomplete: {key}")
    splits = manifest.get("splits", {})
    for name in ("train", "model_validation", "calibration", "test"):
        if not splits.get(name):
            blockers.append(f"Dataset split is missing: {name}")

    if not (METHOD / "scripts/train.py").is_file():
        blockers.append("Multimodal grouped-question trainer is not implemented")
    if not (METHOD / "scripts/predict.py").is_file():
        blockers.append("Checkpoint prediction/reload adapter is not implemented")
    if not (METHOD / "scripts/validate_data.py").is_file():
        blockers.append("Strict split/provenance leakage validator is not implemented")
    elif manifest.get("status") == "ready":
        audit = subprocess.run([sys.executable, str(METHOD / "scripts/validate_data.py")],
                               capture_output=True, text=True, timeout=300)
        if audit.returncode:
            blockers.append("Full data/leakage audit failed: " + (audit.stderr.strip() or audit.stdout.strip()))
        else:
            print(audit.stdout.strip())

    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            blockers.append("Current Python environment has no usable CUDA PyTorch (check CUDA build/driver compatibility)")
        else:
            device = torch.cuda.get_device_properties(0)
            vram_gib = device.total_memory / (1024**3)
            print(f"GPU: {device.name}, {vram_gib:.1f} GiB VRAM")
            if vram_gib < 23:
                blockers.append(f"GPU has {vram_gib:.1f} GiB runtime-visible VRAM; M14 V1 launch gate is 23 GiB")
        try:
            import torchvision  # type: ignore
            import PIL  # type: ignore
            print(f"Image runtime: torchvision {torchvision.__version__}, Pillow {PIL.__version__}")
        except ImportError as exc:
            blockers.append(f"Image runtime dependency is missing: {exc}")
    except ImportError:
        blockers.append("PyTorch is not installed in the current Python environment")

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
        name, memory_mib = [part.strip() for part in gpu.split(",", maxsplit=1)]
        memory_gib = int(memory_mib) / 1024
        print(f"Detected GPU: {name}, {memory_gib:.1f} GiB VRAM")
        if memory_gib < 23:
            blockers.append(f"Detected GPU has {memory_gib:.1f} GiB runtime-visible VRAM; M14 V1 launch gate is 23 GiB")
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        blockers.append("Could not verify GPU model and VRAM with nvidia-smi")

    if blockers:
        print("M14 is not ready to train:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 2
    print("M14 preflight passed. This script does not start training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
