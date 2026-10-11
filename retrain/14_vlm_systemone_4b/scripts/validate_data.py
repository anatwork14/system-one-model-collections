#!/usr/bin/env python3
"""Strict M14 split/provenance audit. Training and calibration use this gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

METHOD = Path(__file__).resolve().parents[1]
DATA = METHOD / "data"
REQUIRED_SPLIT_NAMES = ("train", "model_validation", "calibration", "test")
OPTIONAL_SPLIT_NAMES = ("ood",)
SPLIT_NAMES = (*REQUIRED_SPLIT_NAMES, *OPTIONAL_SPLIT_NAMES)
TEXT_MIN_CHARS = 128
DHASH_MAX_DISTANCE = 2


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def dhash(path: Path) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the image leakage audit; install Pillow before preparing M14 data") from exc
    with Image.open(path) as source:
        image = source.convert("L").resize((9, 8))
        pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def row_text_key(row: dict[str, Any]) -> str | None:
    state = row.get("state", "")
    state_text = state if isinstance(state, str) else canonical(state)
    if len(state_text.strip()) < TEXT_MIN_CHARS:
        return None  # short image-only prompts are commonly shared templates
    candidates = sorted((str(c["id"]), " ".join(str(c["text"]).split())) for c in row["candidates"])
    payload = {"state": " ".join(state_text.split()),
               "question": " ".join(str(row["question"]).split()),
               "type": row["type"], "candidates": candidates}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def validate_record(row: dict[str, Any], split: str, line: int) -> None:
    required = ("id", "source", "source_family", "source_record_id", "group_id", "images",
                "state", "question", "type", "candidates", "target")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{split}:{line}: missing fields {missing}")
    if not all(isinstance(row[k], str) and row[k].strip() for k in
               ("id", "source", "source_family", "source_record_id", "group_id", "question", "type")):
        raise ValueError(f"{split}:{line}: identity and question fields must be nonempty strings")
    if not isinstance(row["images"], list) or not row["images"] or not all(isinstance(x, str) and x for x in row["images"]):
        raise ValueError(f"{split}:{line}: images must be a nonempty list of relative paths")
    if not isinstance(row["candidates"], list) or len(row["candidates"]) < 2:
        raise ValueError(f"{split}:{line}: candidates must contain at least two options")
    candidate_ids = [c.get("id") for c in row["candidates"] if isinstance(c, dict)]
    if len(candidate_ids) != len(row["candidates"]) or any(not isinstance(x, str) or not x for x in candidate_ids):
        raise ValueError(f"{split}:{line}: every candidate needs a nonempty string id")
    if len(candidate_ids) != len(set(candidate_ids)) or row["target"] not in candidate_ids:
        raise ValueError(f"{split}:{line}: candidate IDs must be unique and include target")
    if any(not isinstance(c.get("text"), str) or not c["text"].strip() for c in row["candidates"]):
        raise ValueError(f"{split}:{line}: candidate text must be nonempty")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifests/manifest.json")
    parser.add_argument("--training-only", action="store_true",
                        help="require and validate train/model_validation; audit all partitions if present")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready":
        raise SystemExit("M14 data manifest is not ready; training is blocked")
    checks = manifest.get("required_checks", {})
    required_checks = ("image_use_terms_verified", "image_provenance_recorded",
                       "test_excluded_from_training_and_model_selection")
    false_checks = [key for key in required_checks if checks.get(key) is not True]
    if false_checks:
        raise SystemExit(f"M14 manifest has incomplete safety checks: {false_checks}")
    sources = manifest.get("sources")
    if not sources:
        raise SystemExit("M14 manifest has no source provenance entries")
    source_names = set()
    for source in sources:
        if not all(source.get(key) for key in ("source", "revision", "source_url", "terms_url", "license", "source_sha256")):
            raise SystemExit("Every source needs immutable revision, source/terms URLs, a license identifier, and source SHA-256")
        if str(source["revision"]).lower() in {"main", "master", "latest", "head"}:
            raise SystemExit(f"Source revision is mutable: {source['source']}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(source["source_sha256"])):
            raise SystemExit(f"Source needs a full 64-character SHA-256: {source['source']}")
        if source.get("image_rights_verified") is not True or source.get("terms_verified") is not True:
            raise SystemExit(f"Source terms are not explicitly verified: {source.get('source')}")
        source_names.add(source["source"])

    splits = manifest.get("splits", {})
    must_exist = ("train", "model_validation") if args.training_only else REQUIRED_SPLIT_NAMES
    if any(not splits.get(name) for name in must_exist):
        raise SystemExit(f"Required M14 splits are missing: {[name for name in must_exist if not splits.get(name)]}")

    manifest_path = args.manifest.resolve()
    audit_path = DATA / "manifests/leakage_audit.json"
    if args.training_only:
        if not audit_path.is_file():
            raise SystemExit("Full split leakage audit is missing; run validate_data.py before training")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed" or audit.get("manifest_sha256") != sha256_file(manifest_path):
            raise SystemExit("Full split leakage audit is stale or failed")
        for split, spec in splits.items():
            if spec and audit.get("split_hashes", {}).get(split) != spec.get("sha256"):
                raise SystemExit(f"Full leakage audit is stale for split: {split}")
        for relative, expected in audit.get("image_files", {}).items():
            image_path = (DATA / relative).resolve()
            if not is_within(image_path, DATA.resolve()) or not image_path.is_file() or sha256_file(image_path) != expected["sha256"]:
                raise SystemExit(f"Image corpus changed since the full leakage audit: {relative}")

    seen_ids: dict[str, str] = {}
    seen_sources: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    seen_families: dict[str, set[str]] = defaultdict(set)
    exact_images: dict[str, tuple[str, str]] = {}
    image_files: dict[str, dict[str, str]] = {}
    perceptual_images: list[tuple[int, str, str]] = []
    perceptual_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    long_text: dict[str, tuple[str, str]] = {}
    summary = {}
    audit_splits = ("train", "model_validation") if args.training_only else SPLIT_NAMES
    for split in audit_splits:
        spec = splits.get(split)
        if not spec:
            continue
        path = (DATA / spec["path"]).resolve()
        if not is_within(path, DATA.resolve()):
            raise SystemExit(f"Split path escapes M14 data directory: {spec['path']}")
        if not path.is_file():
            raise SystemExit(f"Split file is missing: {path}")
        actual_sha = sha256_file(path)
        if spec.get("sha256") != actual_sha:
            raise SystemExit(f"Split hash does not match manifest: {split}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or spec.get("rows") != len(rows):
            raise SystemExit(f"Split count is empty or does not match manifest: {split}")
        for line_no, row in enumerate(rows, 1):
            validate_record(row, split, line_no)
            if row["source"] not in source_names:
                raise SystemExit(f"{split}:{line_no}: source is absent from the provenance manifest")
            source_record_key = f"{row['source']}:{row['source_record_id']}"
            for key, registry, allow_same_split in (
                (row["id"], seen_ids, False), (source_record_key, seen_sources, False),
                (f"{row['source']}:{row['group_id']}", seen_groups, True),
            ):
                if key in registry and registry[key] != split:
                    raise SystemExit(f"Leakage: {key} occurs in {registry[key]} and {split}")
                if key in registry and not allow_same_split:
                    raise SystemExit(f"Duplicate record identity in {split}: {key}")
                registry[key] = split
            seen_families[f"{row['source']}:{row['source_family']}"].add(split)
            text_key = row_text_key(row)
            if text_key:
                if text_key in long_text and long_text[text_key][0] != split:
                    raise SystemExit(f"Leakage: repeated long state/question/candidate text in {long_text[text_key][0]} and {split}")
                long_text[text_key] = (split, row["id"])
            for relative in row["images"]:
                image_path = (DATA / relative).resolve()
                if not is_within(image_path, DATA.resolve()) or not image_path.is_file():
                    raise SystemExit(f"Image is missing or escapes data directory: {relative}")
                image_hash = sha256_file(image_path)
                if image_hash in exact_images and exact_images[image_hash][0] != split:
                    raise SystemExit(f"Leakage: exact image duplicate across {exact_images[image_hash][0]} and {split}")
                exact_images[image_hash] = (split, relative)
                image_dhash = dhash(image_path)
                if relative in image_files and image_files[relative]["split"] != split:
                    raise SystemExit(f"Leakage: same image path is assigned to multiple splits: {relative}")
                image_files[relative] = {"sha256": image_hash, "dhash": f"{image_dhash:016x}", "split": split}
                possible_prior = set()
                for band in range(4):
                    possible_prior.update(perceptual_buckets[(band, (image_dhash >> (16 * band)) & 0xFFFF)])
                for prior_index in possible_prior:
                    prior_hash, prior_split, prior_path = perceptual_images[prior_index]
                    if prior_split != split and bin(image_dhash ^ prior_hash).count("1") <= DHASH_MAX_DISTANCE:
                        raise SystemExit(f"Leakage: perceptual image duplicate across {prior_split} ({prior_path}) and {split} ({relative})")
                prior_index = len(perceptual_images)
                perceptual_images.append((image_dhash, split, relative))
                for band in range(4):
                    perceptual_buckets[(band, (image_dhash >> (16 * band)) & 0xFFFF)].add(prior_index)
        summary[split] = {"rows": len(rows), "sha256": actual_sha}
    for family, partitions in seen_families.items():
        if "ood" in partitions and len(partitions) > 1:
            raise SystemExit(f"Leakage: OOD source family also occurs in an in-domain split: {family}")
    result = {"status": "ready", "splits": summary,
                      "unique_ids": len(seen_ids), "unique_source_records": len(seen_sources),
                      "unique_groups": len(seen_groups), "unique_families": len(seen_families),
                      "unique_images": len(exact_images)}
    if not args.training_only:
        computed_checks = {"group_disjoint_splits": True,
                           "image_exact_and_perceptual_deduplication": True,
                           "exact_text_and_source_group_leakage_checks": True,
                           "checksums_recorded": True,
                           "test_excluded_from_training_and_model_selection": True}
        audit = {"format": 1, "status": "passed", "manifest_sha256": sha256_file(manifest_path),
                 "split_hashes": {name: spec["sha256"] for name, spec in splits.items() if spec},
                 "image_files": image_files,
                 "checks": computed_checks,
                 "counts": {name: row["rows"] for name, row in summary.items()}}
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        result["leakage_audit"] = str(audit_path.relative_to(METHOD))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
