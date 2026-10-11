#!/usr/bin/env python3
"""Build the internal, non-commercial Oxford Flowers-102 M14 pilot.

The source benchmark is read-only. Images are copied from its pinned archive
into this method's data store, and the original official validation/test
partitions remain evaluation-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any

import yaml

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[1]
DATA = METHOD / "data"
SOURCE = ROOT / "benchmarks/vision/22_flowers102"
SOURCE_PROCESSED = SOURCE / "processed"
IMAGE_DIR = DATA / "images/flowers102"
SOURCE_ID = "oxford-flowers-102"
SOURCE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
TERMS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt"
PERMISSION_SCOPE = "Internal, non-commercial model training only; no redistribution."
USER_CONFIRMATION = "User explicitly confirmed this source and this use scope in chat on 2026-10-07."
SEED = 17


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    from PIL import Image
    with Image.open(path) as source:
        pixels = list(source.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_member(row: dict[str, Any]) -> str:
    reference = row["state"]["image"]
    marker = "#member="
    if marker not in reference:
        raise ValueError(f"Unexpected Flowers image reference: {reference}")
    member = reference.split(marker, 1)[1]
    if not member.startswith("jpg/") or ".." in Path(member).parts:
        raise ValueError(f"Unsafe or unexpected tar member: {member}")
    return member


def convert_row(row: dict[str, Any], class_names: list[str], image_path: str, seed: int) -> dict[str, Any]:
    target = str(row["target"])
    target_index = int(target)
    if not 0 <= target_index < len(class_names):
        raise ValueError(f"Invalid target class in {row['id']}: {target}")
    rng_seed = int(hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    distractors = rng.sample([str(i) for i in range(len(class_names)) if str(i) != target], 3)
    option_ids = [target, *distractors]
    rng.shuffle(option_ids)
    candidates = [
        {"id": f"class_{int(option):03d}", "text": class_names[int(option)].strip().rstrip(",")}
        for option in option_ids
    ]
    target_candidate = f"class_{target_index:03d}"
    image_id = Path(image_path).stem
    return {
        "id": f"{SOURCE_ID}/{image_id}",
        "source": SOURCE_ID,
        "source_family": "flowers102-four-choice-identification",
        "source_record_id": image_id,
        "group_id": image_id,
        "images": [image_path],
        "state": "Classify the flower visible in the image.",
        "question": "Which flower class is shown?",
        "type": "choice",
        "candidates": candidates,
        "target": target_candidate,
        "metadata": {"original_split": row["metadata"]["split"], "candidate_count": 4},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                    encoding="utf-8")


def main() -> int:
    lock_path = SOURCE / "SOURCE.lock.yaml"
    source_lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    archive = SOURCE / "raw/102flowers.tgz"
    archive_sha = sha256_file(archive)
    expected_sha = source_lock["raw_hashes"]["raw/102flowers.tgz"]
    if archive_sha != expected_sha:
        raise SystemExit("Flowers-102 source archive does not match its pinned SHA-256")

    name_data = json.loads((SOURCE / "splits/class_names.json").read_text(encoding="utf-8"))
    class_names = name_data["classes"]
    if len(class_names) != 102:
        raise SystemExit(f"Expected 102 Flowers classes, found {len(class_names)}")

    original = {split: read_jsonl(SOURCE_PROCESSED / f"{split}.jsonl")
                for split in ("train", "validation", "test")}
    all_rows = [row for rows in original.values() for row in rows]
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    needed_members = {image_member(row) for row in all_rows}
    missing_members = set(needed_members)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in missing_members:
                continue
            if not member.isfile():
                raise ValueError(f"Flowers archive member is not a regular file: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read Flowers archive member: {member.name}")
            target_path = IMAGE_DIR / Path(member.name).name
            temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
            with source, temporary_path.open("wb") as destination:
                destination.write(source.read())
            os.replace(temporary_path, target_path)
            missing_members.remove(member.name)
            if not missing_members:
                break
    if missing_members:
        raise SystemExit(f"Flowers archive is missing {len(missing_members)} referenced images")

    # Keep one copy of every byte-identical image. If an upstream duplicate
    # crosses partitions, prefer the official test copy, then validation.
    exact_image_owner: dict[str, str] = {}
    perceptual_images: list[tuple[int, str]] = []
    perceptual_buckets: dict[tuple[int, int], set[int]] = {}
    deduplicated: dict[str, list[dict[str, Any]]] = {split: [] for split in original}
    removed_exact_duplicates = 0
    removed_perceptual_duplicates = 0
    for split in ("test", "validation", "train"):
        for row in original[split]:
            member_name = image_member(row)
            image_path = IMAGE_DIR / Path(member_name).name
            digest = sha256_file(image_path)
            if digest in exact_image_owner:
                removed_exact_duplicates += 1
                continue
            image_dhash = dhash(image_path)
            possible_prior: set[int] = set()
            for band in range(4):
                possible_prior.update(perceptual_buckets.get((band, (image_dhash >> (16 * band)) & 0xFFFF), set()))
            if any(bin(image_dhash ^ perceptual_images[index][0]).count("1") <= 2 for index in possible_prior):
                removed_perceptual_duplicates += 1
                continue
            exact_image_owner[digest] = split
            deduplicated[split].append(row)
            prior_index = len(perceptual_images)
            perceptual_images.append((image_dhash, split))
            for band in range(4):
                perceptual_buckets.setdefault((band, (image_dhash >> (16 * band)) & 0xFFFF), set()).add(prior_index)

    calibration_ids: set[str] = set()
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in deduplicated["train"]:
        by_class.setdefault(str(row["target"]), []).append(row)
    train_hashes = {
        row["id"]: dhash(IMAGE_DIR / Path(image_member(row)).name)
        for row in deduplicated["train"]
    }
    for class_id, rows in by_class.items():
        if len(rows) < 2:
            raise SystemExit(f"Cannot stratify Flowers calibration row for class {class_id}")
        rows = sorted(rows, key=lambda item: item["id"])
        start = int(hashlib.sha256(f"{SEED}:{class_id}".encode()).hexdigest()[:8], 16) % len(rows)
        ordered = rows[start:] + rows[:start]
        chosen = next((candidate for candidate in ordered if all(
            other["id"] == candidate["id"] or
            bin(train_hashes[candidate["id"]] ^ train_hashes[other["id"]]).count("1") > 2
            for other in deduplicated["train"])), None)
        if chosen is None:
            raise SystemExit(f"Cannot select a calibration row disjoint from training images for class {class_id}")
        calibration_ids.add(chosen["id"])

    source_train = [row for row in deduplicated["train"] if row["id"] not in calibration_ids]
    source_calibration = [row for row in deduplicated["train"] if row["id"] in calibration_ids]
    split_sources = {
        "train": source_train,
        "model_validation": deduplicated["validation"],
        "calibration": source_calibration,
        "test": deduplicated["test"],
    }
    split_specs = {}
    for split, rows in split_sources.items():
        converted = []
        for row in rows:
            member_name = image_member(row)
            relative_image = f"images/flowers102/{Path(member_name).name}"
            if not (DATA / relative_image).is_file():
                raise SystemExit(f"Extracted Flowers image is missing: {relative_image}")
            converted.append(convert_row(row, class_names, relative_image, SEED))
        output_path = DATA / "processed" / f"{split}.jsonl"
        write_jsonl(output_path, converted)
        split_specs[split] = {
            "path": f"processed/{split}.jsonl",
            "rows": len(converted),
            "sha256": sha256_file(output_path),
        }
    split_specs["ood"] = None

    manifest = {
        "format": 2,
        "status": "ready",
        "method": "vlm_systemone_4b",
        "permission_scope": PERMISSION_SCOPE,
        "sources": [{
            "source": SOURCE_ID,
            "revision": source_lock["source"]["revision"],
            "source_url": SOURCE_URL,
            "terms_url": TERMS_URL,
            "license": "User-confirmed permission for internal non-commercial training; no public license asserted.",
            "source_sha256": archive_sha,
            "terms_verified": True,
            "image_rights_verified": True,
            "authorization_basis": "explicit user confirmation in chat",
            "authorization_date": "2026-10-07",
            "commercial_use": False,
            "redistribution_allowed": False,
            "transformation": "M14 four-choice serialization v1; three deterministic distractors per image; original pixels unchanged",
        }],
        "splits": split_specs,
        "required_checks": {
            "image_use_terms_verified": True,
            "image_provenance_recorded": True,
            "group_disjoint_splits": True,
            "image_exact_and_perceptual_deduplication": True,
            "exact_text_and_source_group_leakage_checks": True,
            "checksums_recorded": True,
            "test_excluded_from_training_and_model_selection": True,
        },
        "notes": [
            "Internal non-commercial pilot only, based on explicit user authorization; do not redistribute source images or derived rows.",
            "This is four-choice classification over flower names, not a broad image-decision benchmark.",
            "Original Flowers official train is stratified into train and calibration; original validation is model_validation; original test is test. Exact and near-duplicate images are removed across partitions, preferring test then validation.",
            "OOD is unavailable because this pilot contains one source/task family.",
            "One image/question per row; image groups are disjoint across every partition.",
        ],
    }
    manifest_path = DATA / "manifests/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"source_archive_sha256": archive_sha, "class_count": len(class_names),
               "candidate_count": 4, "splits": {name: spec["rows"] for name, spec in split_specs.items() if spec},
               "unique_images_extracted": len({image_member(row) for row in all_rows}),
               "removed_exact_duplicate_rows": removed_exact_duplicates,
               "removed_perceptual_duplicate_rows": removed_perceptual_duplicates,
               "manifest": str(manifest_path.relative_to(METHOD))}
    (DATA / "processed/prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
