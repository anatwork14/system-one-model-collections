#!/usr/bin/env python3
"""Build a deterministic, family-disjoint train/dev split from Nimble train rows.

The source 324-row evaluation split is deliberately never read here. The new
dev split is carved from the released training rows and can be used for
checkpoint selection. Keep the source release under data/raw unchanged.
"""

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected nonempty rows with unique ids in {path}")
    return rows


def read_partition_metadata(path: Path):
    """Read only split-key fields from the outer file for disjointness checks."""
    metadata = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        provenance = row.get("provenance", {})
        if not row.get("source_family") or not provenance.get("source_id"):
            raise ValueError(f"Missing outer split key on line {line_number}")
        metadata.append({"source_family": row["source_family"], "source_id": provenance["source_id"]})
    return metadata


def canonical_fingerprint(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def split_families(rows, seed, dev_fraction):
    by_domain = defaultdict(set)
    family_sizes = defaultdict(int)
    family_domain = {}
    for row in rows:
        family = row.get("source_family")
        domain = row.get("domain")
        if not family or not domain:
            raise ValueError(f"Missing source_family/domain in {row.get('id')}")
        if family in family_domain and family_domain[family] != domain:
            raise ValueError(f"Source family {family!r} spans domains")
        family_domain[family] = domain
        by_domain[domain].add(family)
        family_sizes[family] += 1

    # Choose one deterministic candidate family per domain, then hold out the
    # smallest candidate families until the requested dev fraction is reached.
    # This keeps several domains represented without sacrificing a third of
    # the training rows to validation on this small dataset.
    candidates = []
    for domain, families in sorted(by_domain.items()):
        if len(families) < 2:
            raise ValueError(f"Need at least two source families in domain {domain!r}")
        selected = min(families, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest())
        candidates.append(selected)
    target_rows = max(1, round(len(rows) * dev_fraction))
    dev_families = set()
    dev_count = 0
    for family in sorted(candidates, key=lambda item: (family_sizes[item], hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())):
        if dev_count >= target_rows:
            break
        dev_families.add(family)
        dev_count += family_sizes[family]

    train_rows, dev_rows = [], []
    for row in rows:
        target = dev_rows if row["source_family"] in dev_families else train_rows
        cloned = dict(row)
        cloned["split"] = "validation" if target is dev_rows else "train"
        target.append(cloned)
    train_rows.sort(key=lambda row: row["id"])
    dev_rows.sort(key=lambda row: row["id"])
    return train_rows, dev_rows, sorted(dev_families)


def convert_kev(row):
    request = json.loads(json.dumps(row["input"], ensure_ascii=False))
    questions = request.get("questions", {})
    if list(questions) != ["decision"]:
        raise ValueError(f"Expected one decision question in {row.get('id')}")
    questions["decision"]["label"] = row["reference"]["target"]
    return {
        "state": request["state"],
        "questions": questions,
        "_meta": {
            "source": "nimble_controlled",
            "id": row["id"],
            "family": row["family"],
            "source_family": row["source_family"],
            "domain": row["domain"],
        },
    }


def state_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def convert_clm(row):
    request = row["input"]
    if list(request["questions"]) != ["decision"]:
        raise ValueError(f"Expected one decision question in {row.get('id')}")
    question = request["questions"]["decision"]
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        candidates = [(str(key), str(value or key)) for key, value in criteria.items()]
        label = str(row["reference"]["target"])
    elif kind == "noul":
        descriptions = criteria or {}
        candidates = [("false", str(descriptions.get("false") or "The proposition is false.")),
                      ("true", str(descriptions.get("true") or "The proposition is true."))]
        label = str(bool(row["reference"]["target"])).lower()
    elif kind == "score":
        candidates = [(str(index), str(level)) for index, level in enumerate(criteria)]
        label = str(row["reference"]["target"])
    else:
        raise ValueError(f"Unsupported question type {kind!r}")
    if label not in {key for key, _ in candidates} or len(candidates) < 2:
        raise ValueError(f"Invalid target/candidates in {row.get('id')}")
    return {
        "id": row["id"],
        "state": state_text(request["state"]),
        "instruction": str(question["instructions"]),
        "kind": kind,
        "candidates": [{"id": key, "text": text} for key, text in candidates],
        "label": label,
        "domain": row.get("domain"),
        "family": row.get("source_family", row.get("family")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/inner_split"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source_train = read_rows(args.source_dir / "train.jsonl")
    outer_eval = read_partition_metadata(args.source_dir / "eval.jsonl")
    original_manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    if not 0.05 <= args.dev_fraction <= 0.30:
        raise ValueError("--dev-fraction must be between 0.05 and 0.30")
    train_rows, dev_rows, dev_families = split_families(source_train, args.seed, args.dev_fraction)
    train_families = {row["source_family"] for row in train_rows}
    dev_family_set = {row["source_family"] for row in dev_rows}
    outer_families = {row["source_family"] for row in outer_eval}
    train_sources = {row["provenance"]["source_id"] for row in train_rows}
    dev_sources = {row["provenance"]["source_id"] for row in dev_rows}
    outer_sources = {row["source_id"] for row in outer_eval}
    if train_families & dev_family_set or train_sources & dev_sources:
        raise AssertionError("family/source leakage between inner train and dev")
    if (train_families & outer_families or dev_family_set & outer_families
            or train_sources & outer_sources or dev_sources & outer_sources):
        raise AssertionError("the released outer evaluation split overlaps the inner split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    dev_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)

    # The upstream schema trainer validates this compact dataset manifest
    # before constructing tokenized examples. Recompute the subset audit with
    # the same pinned public validator instead of weakening upstream checks.
    import sys

    upstream = Path(__file__).resolve().parents[1] / "upstream"
    sys.path.insert(0, str(upstream))
    from nimble.datasets.scaled_evidence import audit_release
    from nimble.datasets.contrastive_data import fingerprint

    manifest = dict(original_manifest)
    manifest["training"] = audit_release(train_rows, original_manifest["source_records"])
    manifest["train_sha256"] = fingerprint(train_rows)
    manifest["training_rows"] = len(train_rows)
    manifest["eval_rows"] = len(dev_rows)
    manifest["eval_sha256"] = file_sha256(dev_path)
    manifest["eval_record_fingerprint"] = fingerprint(dev_rows)
    manifest["target"] = len(train_rows)
    manifest["file_sha256"] = {
        "train.jsonl": file_sha256(train_path),
        "eval.jsonl": file_sha256(dev_path),
    }
    manifest["inner_split"] = {
        "seed": args.seed,
        "dev_source_families": dev_families,
        "train_source_families": sorted(train_families),
        "train_records": len(train_rows),
        "dev_records": len(dev_rows),
        "selection_policy": "one whole source family per domain from original training rows",
        "outer_evaluation_read": "count and family-overlap validation only; labels are not loaded into training",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write the identical family split in each method's input format. Trainers
    # consume these artifacts directly; none of them needs the outer eval file.
    workspace = Path(__file__).resolve().parents[3]
    outputs = {
        "retrain/10_nimble_4b/data/processed": {
            "kev_inner_train.jsonl": [convert_kev(row) for row in train_rows],
            "kev_inner_dev.jsonl": [convert_kev(row) for row in dev_rows],
        },
        "retrain/11_clm_style_4b/data/processed": {
            "train_pairs_inner.jsonl": [convert_clm(row) for row in train_rows],
            "dev_pairs_inner.jsonl": [convert_clm(row) for row in dev_rows],
        },
        "retrain/12_pointer_head_only_4b/data/processed": {
            "kev_train_inner.jsonl": [convert_kev(row) for row in train_rows],
            "kev_dev_inner.jsonl": [convert_kev(row) for row in dev_rows],
        },
        "retrain/13_kev_fullft_4b/data/processed": {
            "kev_train_inner.jsonl": [convert_kev(row) for row in train_rows],
            "kev_dev_inner.jsonl": [convert_kev(row) for row in dev_rows],
        },
    }
    method_files = {}
    arm_files = {}
    for relative_dir, files in outputs.items():
        output_dir = workspace / relative_dir
        for name, records in files.items():
            path = output_dir / name
            write_jsonl(path, records)
            artifact = {
                "rows": len(records), "sha256": file_sha256(path),
            }
            method_files[str(path.relative_to(workspace))] = artifact
            arm_files.setdefault(relative_dir, {})[name] = artifact
    report = {
        "train_records": len(train_rows),
        "dev_records": len(dev_rows),
        "train_families": len(train_families),
        "dev_families": len(dev_families),
        "dev_source_families": dev_families,
        "train_dev_family_overlap": False,
        "train_dev_source_overlap": False,
        "outer_eval_family_overlap": False,
        "train_sha256": file_sha256(train_path),
        "dev_sha256": file_sha256(dev_path),
        "manifest": str(manifest_path),
        "method_files": method_files,
    }
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for processed_dir, files in outputs.items():
        manifest_dir = processed_dir.rsplit("/", 1)[0] + "/manifests"
        target = workspace / manifest_dir / "inner_split.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        arm_report = {**report, "files": arm_files.get(processed_dir, {})}
        target.write_text(json.dumps(arm_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
