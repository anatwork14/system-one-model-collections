#!/usr/bin/env python3
"""Download only the official Clef-Flash architecture and tokenizer artifacts."""

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
FILES = [
    "README.md",
    "LICENSE",
    "joint_schema_model.py",
    "joint_head_config.json",
    "joint_head.safetensors",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


if __name__ == "__main__":
    target = ROOT / "weights" / "architecture"
    snapshot_download(
        repo_id="Cloudflare/clef-flash",
        revision=REVISION,
        local_dir=target,
        allow_patterns=FILES,
    )
    print(f"Downloaded pinned architecture files to {target}")
