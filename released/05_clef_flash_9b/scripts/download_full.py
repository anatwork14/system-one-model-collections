#!/usr/bin/env python3
"""Download the complete pinned official Cloudflare Clef-Flash release."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
FILES = [
    "*.safetensors",
    "*.json",
    "*.py",
    "README.md",
    "LICENSE",
    "chat_template.jinja",
    ".gitattributes",
]


if __name__ == "__main__":
    target = ROOT / "weights" / "model"
    snapshot_download(
        repo_id="Cloudflare/clef-flash",
        revision=REVISION,
        local_dir=target,
        allow_patterns=FILES,
    )
    print(f"Downloaded the complete pinned Clef-Flash release to {target}")
