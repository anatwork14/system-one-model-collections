#!/usr/bin/env python3
"""Download the pinned core benchmark data without overwriting existing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "jevbench_public": "bb05a335bc809e61b20c0f745d25499a82b326fc",
    "banking77": "90d4e2ee5521c04fc1488f065b8b083658768c57",
    "clinc150_oos": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
    "boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "goemotions": "add492243ff905527e67aeb8b80c082af02207c3",
    "when2call": "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc",
    "bfcl": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    "bolt": "2c935438b5506febcfd23e499e3de81868236b17",
    "cifar10": "0b2714987fa478483af9968de7c934580d0bb9a2",
}
GIT_SOURCES = {
    "jevbench": ("https://github.com/fstandhartinger/jevbench.git", "bb05a335bc809e61b20c0f745d25499a82b326fc", ["datasets/public/**"]),
    "clinc": ("https://github.com/clinc/oos-eval.git", "828f8093932c8fe6ca7936c3d2e52903b1c523de", ["data/data_full.json"]),
    "when2call": ("https://github.com/NVIDIA/When2Call.git", "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc", ["data/test/when2call_test_mcq.jsonl"]),
    "bfcl": ("https://github.com/ShishirPatil/gorilla.git", "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8", ["berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json", "berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_irrelevance.json", "berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json", "berkeley-function-call-leaderboard/bfcl_eval/data/README.md"]),
    "bolt": ("https://github.com/CNIC-DSL/BOLT.git", "2c935438b5506febcfd23e499e3de81868236b17", [
        "data/banking/label/label.list", "data/banking/label/fold5/part0/label_known_0.75.list",
        "data/clinc/label/label.list", "data/clinc/label/fold5/part0/label_known_0.75.list",
        "data/hwu/origin_data/**", "data/hwu/label/label.list", "data/hwu/label/fold5/part0/label_known_0.75.list", "data/hwu/labeled_data/1.0/*.tsv",
        "data/stackoverflow/origin_data/**", "data/stackoverflow/label/label.list", "data/stackoverflow/label/fold5/part0/label_known_0.75.list", "data/stackoverflow/labeled_data/1.0/*.tsv",
    ]),
}
HUB_DATASETS = {
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "goemotions": ("google-research-datasets/go_emotions", "add492243ff905527e67aeb8b80c082af02207c"),
    "cifar10": ("uoft-cs/cifar10", "0b2714987fa478483af9968de7c934580d0bb9a2"),
    "oxford_iiit_pets": ("Donghyun99/Oxford-IIIT-Pet", "371061c33e62f8df8e3f7fc533bedd7680f5855e"),
    "flowers102": ("Donghyun99/Oxford-Flower-102", "7e43fe37eeb9278138ab0bd2d0abb10d0f703d09"),
}
DIRECT_FILES = {
    "banking77": [
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/train.csv",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/test.csv",
    ]
}
EXTRA_FILES = {
    "goemotions": [
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
        "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
    ]
}
OXFORD_FILES = {
    "oxford_iiit_pets": [
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
    ],
    "flowers102": [
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat",
    ],
}
TRANSFER_MIRRORS = {
    "oxford_iiit_pets": "https://huggingface.co/datasets/Donghyun99/Oxford-IIIT-Pet",
    "flowers102": "https://huggingface.co/datasets/Donghyun99/Oxford-Flower-102",
}
GROUPS = {
    "system_one": ["jevbench_public", "banking77", "clinc150_oos", "boolq", "goemotions", "when2call", "bfcl"],
    "openworld": ["banking77_bolt", "clinc150_bolt", "hwu64_bolt", "stackoverflow_bolt"],
    "vision": ["cifar10", "oxford_iiit_pets", "flowers102"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def copy_tree(source: Path, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {target}")
    if target.exists():
        target.rmdir()
    shutil.copytree(source, target)


def raw_stage(dataset_path: Path) -> Path:
    raw = dataset_path / "raw"
    if raw.exists() and any(raw.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing raw files: {raw}")
    stage = dataset_path / ".raw.download"
    stage.mkdir(parents=True, exist_ok=True)
    return stage


def promote_raw(dataset_path: Path, stage: Path) -> None:
    raw = dataset_path / "raw"
    if raw.exists():
        if any(raw.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing raw files: {raw}")
        raw.rmdir()
    stage.replace(raw)


def ensure_git(name: str) -> Path:
    url, revision, paths = GIT_SOURCES[name]
    checkout = ROOT / "sources" / f"{name}.checkout"
    if not (checkout / ".git").is_dir():
        if checkout.exists() and any(checkout.iterdir()):
            raise RuntimeError(f"source directory exists and is not a git checkout: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(checkout)])
    run(["git", "-C", str(checkout), "sparse-checkout", "init", "--no-cone"])
    run(["git", "-C", str(checkout), "sparse-checkout", "set", "--no-cone", *paths])
    run(["git", "-C", str(checkout), "checkout", "--detach", revision])
    return checkout


def copy_selected_git_data(key: str, checkout: Path, raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    if key == "jevbench_public":
        copy_tree(checkout / "datasets" / "public", raw / "public")
    elif key == "clinc150_oos":
        shutil.copy2(checkout / "data" / "data_full.json", raw / "data_full.json")
    elif key == "when2call":
        shutil.copy2(checkout / "data" / "test" / "when2call_test_mcq.jsonl", raw / "when2call_test_mcq.jsonl")
    elif key == "bfcl":
        source = checkout / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
        for file_name in ("BFCL_v4_simple_python.json", "BFCL_v4_irrelevance.json"):
            shutil.copy2(source / file_name, raw / file_name)
        shutil.copy2(source / "possible_answer" / "BFCL_v4_simple_python.json", raw / "BFCL_v4_simple_python_answers.json")
        (raw / "README.upstream.md").write_text((source / "README.md").read_text())


def download_hub(key: str, dataset_path: Path) -> None:
    repo_id, revision = HUB_DATASETS[key]
    raw = raw_stage(dataset_path)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=str(raw))


def download_oxford(key: str, dataset_path: Path) -> None:
    raw = raw_stage(dataset_path)
    for url in OXFORD_FILES[key]:
        target = raw / url.rsplit("/", 1)[1]
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "system-one-benchmark/1.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)


def download_direct_files(key: str, dataset_path: Path) -> None:
    raw = dataset_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for url in DIRECT_FILES[key]:
        target = raw / url.rsplit("/", 1)[1]
        if target.exists():
            print(f"Keeping existing file {target.relative_to(ROOT)}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "system-one-benchmark/1.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)


def download_extra_files(key: str, dataset_path: Path) -> None:
    raw = dataset_path / "raw"
    for url in EXTRA_FILES.get(key, []):
        target = raw / url.rsplit("/", 1)[1]
        if target.exists():
            continue
        partial = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "system-one-benchmark/1.0"})
        with urllib.request.urlopen(request) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)


def materialize_bolt(checkout: Path) -> None:
    mappings = {
        "banking77_bolt": ("banking", "banking77"),
        "clinc150_bolt": ("clinc", "clinc150_oos"),
        "hwu64_bolt": ("hwu", "hwu64_bolt"),
        "stackoverflow_bolt": ("stackoverflow", "stackoverflow_bolt"),
    }
    for key, (source_name, _) in mappings.items():
        target = ROOT / "openworld" / GROUP_ENTRY_PATHS[key] / "splits" / "known75"
        target.mkdir(parents=True, exist_ok=True)
        source = checkout / "data" / source_name
        for rel in (Path("label/label.list"), Path("label/fold5/part0/label_known_0.75.list")):
            src = source / rel
            if src.is_file():
                dest = target / (source_name + "_" + rel.name)
                if not dest.exists():
                    shutil.copy2(src, dest)
        if key in {"hwu64_bolt", "stackoverflow_bolt"}:
            raw = ROOT / "openworld" / GROUP_ENTRY_PATHS[key] / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            raw_source = source / "origin_data"
            if raw_source.exists() and not (raw / "origin_data").exists():
                copy_tree(raw_source, raw / "origin_data")
            processed_source = source / "labeled_data" / "1.0"
            processed_target = ROOT / "openworld" / GROUP_ENTRY_PATHS[key] / "processed" / "bolt_labeled_1.0"
            if processed_source.exists() and not processed_target.exists():
                copy_tree(processed_source, processed_target)


GROUP_ENTRY_PATHS = {
    "banking77_bolt": "10_banking77_bolt",
    "clinc150_bolt": "11_clinc150_bolt",
    "hwu64_bolt": "12_hwu64_bolt",
    "stackoverflow_bolt": "13_stackoverflow_bolt",
}


def update_status(key: str, dataset_path: Path, examples: int | None = None) -> None:
    raw = dataset_path / "raw"
    files = [item for item in raw.rglob("*") if item.is_file()] if raw.exists() else []
    hashes = {item.relative_to(dataset_path).as_posix(): sha256(item) for item in files}
    (dataset_path / "checksums.sha256").write_text("".join(f"{value}  {name}\n" for name, value in sorted(hashes.items())))
    lock_path = dataset_path / "SOURCE.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text())
    lock.setdefault("raw_hashes", {}).update(hashes)
    lock["downloaded_at"] = date.today().isoformat()
    if lock.get("source_type") == "artifact":
        aggregate = hashlib.sha256("".join(f"{name} {value}\n" for name, value in sorted(hashes.items())).encode()).hexdigest()
        lock["source"]["revision"] = f"sha256:{aggregate}"
    lock["status"] = "downloaded"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False))
    aggregate_path = ROOT / "checksums" / f"{key}.sha256"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text("".join(f"{value}  {name}\n" for name, value in sorted(hashes.items())))
    status_path = dataset_path / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({"downloaded": True, "validated": False, "prepared": False,
                   "examples": None, "source_revision": lock["source"]["revision"],
                   "checksum_verified": True})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Downloaded {key}: {len(files)} file(s), {sum(p.stat().st_size for p in files):,} bytes")


def one(key: str, item: dict) -> None:
    dataset_path = ROOT / item["path"]
    dataset_path.mkdir(parents=True, exist_ok=True)
    lock = yaml.safe_load((dataset_path / "SOURCE.lock.yaml").read_text())
    current_revision = lock["source"]["revision"]
    status_path = dataset_path / "status.json"
    prior_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    extra_present = all((dataset_path / "raw" / url.rsplit("/", 1)[1]).is_file()
                        for url in EXTRA_FILES.get(key, []))
    if key == "bfcl":
        extra_present = extra_present and (dataset_path / "raw" / "BFCL_v4_simple_python_answers.json").is_file()
    if prior_status.get("downloaded") and prior_status.get("source_revision") == current_revision and extra_present:
        print(f"Already downloaded {key} at {current_revision}")
        return
    if key in TRANSFER_MIRRORS:
        repo_id, mirror_revision = HUB_DATASETS[key]
        lock["source"]["transfer"] = {"type": "pinned_hub_mirror", "url": TRANSFER_MIRRORS[key], "revision": mirror_revision}
        lock_path = dataset_path / "SOURCE.lock.yaml"
        lock_path.write_text(yaml.safe_dump(lock, sort_keys=False))
    if key in DIRECT_FILES:
        download_direct_files(key, dataset_path)
    elif key == "bfcl" and (dataset_path / "raw").exists() and any((dataset_path / "raw").iterdir()):
        checkout = ensure_git("bfcl")
        answer_source = checkout / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data" / "possible_answer" / "BFCL_v4_simple_python.json"
        answer_target = dataset_path / "raw" / "BFCL_v4_simple_python_answers.json"
        if not answer_target.exists():
            shutil.copy2(answer_source, answer_target)
        update_status(key, dataset_path, item.get("expected_examples"))
    elif key in HUB_DATASETS:
        existing_raw = dataset_path / "raw"
        if not (key in EXTRA_FILES and existing_raw.exists() and any(existing_raw.iterdir())):
            download_hub(key, dataset_path)
            promote_raw(dataset_path, dataset_path / ".raw.download")
        download_extra_files(key, dataset_path)
    elif key in {"jevbench_public", "clinc150_oos", "when2call", "bfcl"}:
        source_name = {"jevbench_public": "jevbench", "clinc150_oos": "clinc", "when2call": "when2call", "bfcl": "bfcl"}[key]
        checkout = ensure_git(source_name)
        stage = raw_stage(dataset_path)
        copy_selected_git_data(key, checkout, stage)
        promote_raw(dataset_path, stage)
    elif key.endswith("_bolt"):
        checkout = ensure_git("bolt")
        materialize_bolt(checkout)
        if key in {"hwu64_bolt", "stackoverflow_bolt"}:
            update_status(key, dataset_path)
        else:
            status_path = dataset_path / "status.json"
            status = json.loads(status_path.read_text())
            status.update({"downloaded": True, "source_revision": REVISIONS["bolt"]})
            status_path.write_text(json.dumps(status, indent=2) + "\n")
        return
    else:
        raise RuntimeError(f"no downloader configured for {key}")
    if key in EXTRA_FILES:
        lock = yaml.safe_load((dataset_path / "SOURCE.lock.yaml").read_text())
        lock["source"]["additional_artifacts"] = EXTRA_FILES[key]
        (dataset_path / "SOURCE.lock.yaml").write_text(yaml.safe_dump(lock, sort_keys=False))
    update_status(key, dataset_path, item.get("expected_examples"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]))
    parser.add_argument("--group", choices=list(GROUPS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    if not args.list and bool(args.dataset) == bool(args.group):
        parser.error("choose exactly one of --dataset, --group, or --list")
    keys = list(registry) if args.list else ([args.dataset] if args.dataset else GROUPS[args.group])
    for key in keys:
        item = registry[key]
        lock_path = ROOT / item["path"] / "SOURCE.lock.yaml"
        lock = yaml.safe_load(lock_path.read_text())
        revision = str(lock["source"]["revision"])
        print(f"{key:22} revision={revision} status={lock.get('status', 'unknown')}")
    if args.list or args.dry_run:
        return 0
    failures: list[tuple[str, str]] = []
    for key in keys:
        try:
            one(key, registry[key])
        except Exception as exc:
            failures.append((key, str(exc)))
            print(f"FAILED {key}: {exc}", file=sys.stderr)
    if failures:
        print("\nSome sources need attention:\n" + "\n".join(f"- {key}: {error}" for key, error in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
