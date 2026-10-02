#!/usr/bin/env python3
"""Record immutable revisions resolved from upstream public remotes."""

from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "jevbench_public": ("git", "bb05a335bc809e61b20c0f745d25499a82b326fc"),
    "banking77": ("git", "57ec275d8078af65b7731c2a98be812d844a6d6b"),
    "clinc150_oos": ("git", "828f8093932c8fe6ca7936c3d2e52903b1c523de"),
    "boolq": ("hub", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "goemotions": ("hub", "add492243ff905527e67aeb8b80c082af02207c3"),
    "when2call": ("git", "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"),
    "bfcl": ("git", "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"),
    "banking77_bolt": ("git", "2c935438b5506febcfd23e499e3de81868236b17"),
    "clinc150_bolt": ("git", "2c935438b5506febcfd23e499e3de81868236b17"),
    "hwu64_bolt": ("git", "2c935438b5506febcfd23e499e3de81868236b17"),
    "stackoverflow_bolt": ("git", "2c935438b5506febcfd23e499e3de81868236b17"),
    "cifar10": ("hub", "0b2714987fa478483af9968de7c934580d0bb9a2"),
    # Oxford source archives use the official download host; artifact hashes
    # are added to the lock immediately after transfer.
    "oxford_iiit_pets": ("artifact", "PIN_REQUIRED_ARTIFACT_SHA256"),
    "flowers102": ("artifact", "PIN_REQUIRED_ARTIFACT_SHA256"),
}

HUB_URLS = {
    "boolq": "https://huggingface.co/datasets/google/boolq",
    "goemotions": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
    "cifar10": "https://huggingface.co/datasets/uoft-cs/cifar10",
}
GIT_URLS = {
    "banking77": "https://github.com/PolyAI-LDN/task-specific-datasets",
    "jevbench_public": "https://github.com/fstandhartinger/jevbench.git",
    "clinc150_oos": "https://github.com/clinc/oos-eval.git",
    "when2call": "https://github.com/NVIDIA/When2Call.git",
    "bfcl": "https://github.com/ShishirPatil/gorilla.git",
    "banking77_bolt": "https://github.com/CNIC-DSL/BOLT.git",
    "clinc150_bolt": "https://github.com/CNIC-DSL/BOLT.git",
    "hwu64_bolt": "https://github.com/CNIC-DSL/BOLT.git",
    "stackoverflow_bolt": "https://github.com/CNIC-DSL/BOLT.git",
}
OFFICIAL_URLS = {
    "oxford_iiit_pets": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    "flowers102": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
}


def main() -> None:
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())
    for key, (kind, revision) in REVISIONS.items():
        item = registry["datasets"][key]
        if key in HUB_URLS:
            item["source"] = HUB_URLS[key]
        elif key in GIT_URLS:
            item["source"] = GIT_URLS[key]
        elif key in OFFICIAL_URLS:
            item["source"] = OFFICIAL_URLS[key]
        lock_path = ROOT / item["path"] / "SOURCE.lock.yaml"
        lock = yaml.safe_load(lock_path.read_text())
        lock["source"]["url"] = item["source"]
        lock["source"]["revision"] = revision
        lock["source_type"] = kind
        lock["status"] = "pinned" if kind != "artifact" else "artifact_hash_pending"
        lock["pinned_at"] = date.today().isoformat()
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False))
    (ROOT / "registry.yaml").write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True))
    print(f"Pinned {len(REVISIONS) - 2} code/hub sources; two official Oxford archives need artifact hashes.")


if __name__ == "__main__":
    main()
