#!/usr/bin/env python3
"""Create family-disjoint canonical Clef records from the frozen Nimble data."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "shared/datasets/nimble"
DATA = Path(__file__).resolve().parents[1] / "data"


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def canonicalize(row: dict) -> dict:
    input_record = row["input"]
    questions = {}
    for key, value in input_record["questions"].items():
        questions[str(key)] = {**value, "target": row["reference"]["target"]}

    atom_by_id = {}
    certificate = row.get("evidence_certificate") or {}
    for atom in certificate.get("spec", {}).get("atoms", []):
        atom_by_id[atom["id"]] = atom["statement"]
    for atom_id, state in certificate.get("fact_states", {}).items():
        if state not in {"supported", "refuted"} or atom_id not in atom_by_id:
            continue
        question_id = "evidence_" + re.sub(r"[^a-z0-9]+", "_", atom_id.lower()).strip("_")
        questions[question_id] = {
            "type": "noul",
            "instructions": f"Is this statement supported by the state? {atom_by_id[atom_id]}",
            "target": state == "supported",
        }
    return {
        "id": row["id"],
        "family": row.get("family", row["id"]),
        "state": input_record["state"],
        "questions": questions,
        "source": "nimble_train",
        "label_status": row.get("quality_status", "source_labeled"),
    }


def family_split(family: str, salt: str) -> str:
    number = int(hashlib.sha256(f"{salt}:{family}".encode()).hexdigest()[:8], 16) % 100
    return "train" if number < 80 else ("calibration" if number < 90 else "dev")


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="clef4b-open-mixture-v1")
    args = parser.parse_args()
    canonical = [canonicalize(row) for row in read_jsonl(SOURCE / "train.jsonl")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in canonical:
        groups[record["family"]].append(record)
    assigned = {family: family_split(family, args.seed) for family in groups}
    splits = {name: [] for name in ("train", "calibration", "dev")}
    for family, records in groups.items():
        splits[assigned[family]].extend(records)
    test = [canonicalize(row) for row in read_jsonl(SOURCE / "eval.jsonl")]
    dump(DATA / "canonical/nimble_canonical.jsonl", canonical)
    for name, rows in splits.items():
        dump(DATA / name / "nimble.jsonl", rows)
    dump(DATA / "test/nimble_frozen.jsonl", test)
    manifest = {
        "dataset": "clef4b-open-mixture-v1-initial",
        "source": "shared/datasets/nimble/{train,eval}.jsonl",
        "split_method": "sha256 family-group assignment; train 80%, calibration 10%, dev 10% of source families",
        "augmentation": "none; perform only after family split",
        "label_caveat": "Nimble records are synthetic and model-checked, not fully human-reviewed. Evidence questions use supported/refuted certificates and omit unknown facts.",
        "counts": {name: len(rows) for name, rows in {**splits, "test": test, "canonical": canonical}.items()},
        "families": len(groups),
        "seed": args.seed,
    }
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    (DATA / "manifests/initial.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
