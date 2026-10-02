#!/usr/bin/env python3
"""Build normalized System-One records and BOLT open-world views."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import tarfile
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
ACTION_DESCRIPTIONS = {
    "direct": "Answer directly without calling a tool.",
    "tool_call": "Call one of the available tools to complete the request.",
    "request_for_info": "Ask the user for missing information before proceeding.",
    "cannot_answer": "Decline or state that the request cannot be answered.",
}


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def record(record_id: str, state: dict, question: dict, target, dataset: str, split: str, **metadata) -> dict:
    return {"id": record_id, "state": state, "question": question, "target": target,
            "metadata": {"dataset": dataset, "split": split, **metadata}}


def choice_question(criteria: dict, instructions="Choose the best matching option.") -> dict:
    return {"id": "class_choice", "type": "choice", "instructions": instructions, "criteria": criteria}


def labels_from_card(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    front = content.split("---", 2)[1]
    metadata = yaml.safe_load(front)
    names = None
    for feature in metadata.get("dataset_info", {}).get("features", []):
        if feature.get("name") == "label":
            names = feature["dtype"]["class_label"]["names"]
            break
    if names is None:
        raise ValueError(f"no class-label names in {path}")
    return [names[str(index)] for index in range(len(names))]


def parquet_rows(path: Path):
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def system_one(key: str) -> dict[str, int]:
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    item = registry[key]
    base = ROOT / item["path"]
    raw, out = base / "raw", base / "processed"
    counts: dict[str, int] = {}

    if key == "jevbench_public":
        for split in ("original", "easy", "hard"):
            source = raw / "public" / f"{split}.jsonl"
            def rows():
                with source.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        state = r.get("state")
                        state = state if isinstance(state, dict) else {"text": str(state)}
                        q = r.get("question", {"type": "choice", "instructions": "Answer the decision."})
                        q = {"id": q.get("id", "decision"), "type": q.get("type", "choice"),
                             "instructions": q.get("instructions", "Answer the decision."), **({"criteria": q["criteria"]} if "criteria" in q else {})}
                        target = r.get("expected", r.get("target"))
                        if q.get("type") == "noul" and target in {"yes", "no"} and "criteria" in q:
                            # JevBench stores a yes/no gold label while the native rubric names
                            # the equivalent criteria keys true/false.
                            target = {"yes": "true", "no": "false"}[target]
                        yield record(r["id"], state, q, target, key, split,
                                     source_group=r.get("group"), family=r.get("family"), provenance=r.get("provenance"))
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key == "banking77":
        labels = sorted({row["category"] for s in ("train", "test") for row in csv.DictReader((raw / f"{s}.csv").open(encoding="utf-8"))})
        criteria = {label: label.replace("_", " ") for label in labels}
        for split in ("train", "test"):
            def rows():
                with (raw / f"{split}.csv").open(encoding="utf-8", newline="") as f:
                    for i, r in enumerate(csv.DictReader(f)):
                        yield record(f"{split}/{i}", {"text": r["text"]}, choice_question(criteria), r["category"], key, split)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key == "clinc150_oos":
        source = json.loads((raw / "data_full.json").read_text())
        labels = sorted({label for split, examples in source.items() for _, label in examples})
        criteria = {label: "Out of scope" if label == "oos" else label.replace("_", " ") for label in labels}
        for split, examples in source.items():
            def rows():
                for i, (text, label) in enumerate(examples):
                    yield record(f"{split}/{i}", {"text": text}, choice_question(criteria), label, key, split)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key == "boolq":
        for split in ("train", "validation"):
            file = next((raw / "data").glob(f"{split}-*.parquet"))
            def rows():
                for i, r in parquet_rows(file):
                    yield record(f"{split}/{i}", {"passage": r["passage"]},
                                 {"id": "boolq", "type": "noul", "instructions": r["question"]},
                                 bool(r["answer"]), key, split)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key == "goemotions":
        for split in ("train", "validation", "test"):
            file = next((raw / "simplified").glob(f"{split}-*.parquet"))
            def rows():
                for _, r in parquet_rows(file):
                    labels = {EMOTIONS[i] for i in r["labels"]}
                    for label in EMOTIONS:
                        yield record(f"{split}/{r['id']}/{label}", {"text": r["text"]},
                                     {"id": f"emotion_{label}", "type": "noul",
                                      "instructions": f"Is this comment expressing {label}?"},
                                     label in labels, key, split, source_id=r["id"], label=label)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key == "when2call":
        criteria = ACTION_DESCRIPTIONS
        def rows():
            with (raw / "when2call_test_mcq.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    yield record(r["uuid"], {"text": r["question"], "tools": r.get("tools", [])},
                                 choice_question(criteria, "Choose the appropriate action for this request."),
                                 r["correct_answer"], key, "test", source_id=r.get("source_id"), source=r.get("source"))
        counts["test"] = write_jsonl(out / "test.jsonl", rows())

    elif key == "bfcl":
        answer_path = raw / "BFCL_v4_simple_python_answers.json"
        answers = {x["id"]: x["ground_truth"] for x in
                   (json.loads(line) for line in answer_path.read_text().splitlines() if line.strip())}
        for split, filename in (("test", "BFCL_v4_simple_python.json"), ("test_irrelevance", "BFCL_v4_irrelevance.json")):
            def rows():
                for line in (raw / filename).read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    messages = ast.literal_eval(r["question"]) if isinstance(r["question"], str) else r["question"]
                    prompt = "\n".join(m.get("content", "") for turn in messages for m in turn if m.get("role") == "user")
                    tools = ast.literal_eval(r["function"]) if isinstance(r["function"], str) else r["function"]
                    criteria = {tool["name"]: tool.get("description", tool["name"]) for tool in tools}
                    if split == "test_irrelevance":
                        criteria["NO_TOOL"] = "No listed tool is relevant to the request."
                        target = "NO_TOOL"
                    else:
                        truth = answers[r["id"]]
                        target = next(iter(truth[0]))
                    yield record(r["id"], {"text": prompt, "tools": tools},
                                 choice_question(criteria, "Choose the single tool that best handles the request."),
                                 target, key, split)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    elif key in {"cifar10", "oxford_iiit_pets", "flowers102"}:
        if key == "oxford_iiit_pets" and (raw / "images.tar.gz").is_file():
            classes = json.loads((base / "splits/class_names.json").read_text())["classes"]
            criteria = {str(i): name for i, name in enumerate(classes)}
            with tarfile.open(raw / "annotations.tar.gz", "r:gz") as annotations:
                for split, member in (("train", "annotations/trainval.txt"), ("test", "annotations/test.txt")):
                    text = annotations.extractfile(member).read().decode("utf-8")
                    def rows(text=text, split=split):
                        for line in text.splitlines():
                            fields = line.split()
                            if not fields or fields[0].startswith("#"):
                                continue
                            image_id, label = fields[0], str(int(fields[1]) - 1)
                            yield record(f"{split}/{image_id}",
                                         {"image": f"raw/images.tar.gz#member=images/{image_id}.jpg"},
                                         choice_question(criteria, "Which pet breed is shown?"), label, key, split)
                    counts[split] = write_jsonl(out / f"{split}.jsonl", rows())
            write_metadata(out / "prepare_summary.json", {"dataset": key, "splits": counts, "schema": "decision_record.v1", "image_storage": "official_tar_members"})
            return counts
        if key == "flowers102" and (raw / "102flowers.tgz").is_file():
            from scipy.io import loadmat
            classes = json.loads((base / "splits/class_names.json").read_text())["classes"]
            criteria = {str(i): name for i, name in enumerate(classes)}
            labels = loadmat(raw / "imagelabels.mat")["labels"].ravel()
            sets = loadmat(raw / "setid.mat")
            split_keys = {"train": "trnid", "validation": "valid", "test": "tstid"}
            for split, mat_key in split_keys.items():
                def rows(split=split, mat_key=mat_key):
                    for image_id in sets[mat_key].ravel().tolist():
                        image_id = int(image_id)
                        label = str(int(labels[image_id - 1]) - 1)
                        member = f"jpg/image_{image_id:05d}.jpg"
                        yield record(f"{split}/image_{image_id:05d}",
                                     {"image": f"raw/102flowers.tgz#member={member}"},
                                     choice_question(criteria, "Which flower class is shown?"), label, key, split)
                counts[split] = write_jsonl(out / f"{split}.jsonl", rows())
            write_metadata(out / "prepare_summary.json", {"dataset": key, "splits": counts, "schema": "decision_record.v1", "image_storage": "official_tar_members"})
            return counts
        class_names = labels_from_card(raw / "README.md")
        splits = ("train", "test") if key != "flowers102" else ("train", "validation", "test")
        for split in splits:
            paths = sorted((raw / "plain_text").glob(f"{split}-*.parquet")) or sorted((raw / "data").glob(f"{split}-*.parquet"))
            if not paths:
                raise FileNotFoundError(f"no image parquet for {key}/{split}")
            def rows():
                for parquet in paths:
                    rel = parquet.relative_to(base).as_posix()
                    for i, r in parquet_rows(parquet):
                        label = int(r["label"])
                        yield record(f"{split}/{parquet.stem}/{i}",
                                     {"image": f"{rel}#row={i}"},
                                     choice_question({str(j): name for j, name in enumerate(class_names)}, "Which class best describes this image?"),
                                     str(label), key, split)
            counts[split] = write_jsonl(out / f"{split}.jsonl", rows())

    else:
        raise ValueError(f"unsupported System-One dataset {key}")
    write_metadata(out / "prepare_summary.json", {"dataset": key, "splits": counts, "schema": "decision_record.v1"})
    return counts


def write_metadata(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def text_examples(key: str) -> dict[str, list[tuple[str, str, str]]]:
    """Return split -> (id, text, label) for BOLT source data, without duplicating raw payloads."""
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    if key == "banking77_bolt":
        path = ROOT / registry["banking77"]["path"] / "raw"
        result = {}
        for split in ("train", "test"):
            with (path / f"{split}.csv").open(encoding="utf-8", newline="") as f:
                result[split] = [(f"{split}/{i}", r["text"], r["category"]) for i, r in enumerate(csv.DictReader(f))]
        return result
    if key == "clinc150_bolt":
        source = json.loads((ROOT / registry["clinc150_oos"]["path"] / "raw/data_full.json").read_text())
        return {split: [(f"{split}/{i}", text, label) for i, (text, label) in enumerate(rows)] for split, rows in source.items()}
    entry = registry[key]
    result = {}
    for split in ("train", "dev", "test"):
        file = ROOT / entry["path"] / "raw/origin_data" / f"{split}.tsv"
        with file.open(encoding="utf-8", newline="") as f:
            result[split] = [(f"{split}/{i}", r["text"], r["label"]) for i, r in enumerate(csv.DictReader(f, delimiter="\t"))]
    return result


def openworld(key: str) -> dict[str, int]:
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    item = registry[key]
    base, out = ROOT / item["path"], ROOT / item["path"] / "processed"
    source_name = {"banking77_bolt": "banking", "clinc150_bolt": "clinc", "hwu64_bolt": "hwu", "stackoverflow_bolt": "stackoverflow"}[key]
    split_dir = base / "splits" / "known75"
    all_labels = [x.strip() for x in (split_dir / f"{source_name}_label.list").read_text().splitlines() if x.strip()]
    known_path = split_dir / f"{source_name}_label_known_0.75.list"
    known = [x.strip() for x in known_path.read_text().splitlines() if x.strip()]
    known_set, unseen_set = set(known), set(all_labels) - set(known)
    write_metadata(split_dir / "known_classes.json", {"dataset": key, "fold": "known75", "classes": known})
    write_metadata(split_dir / "unseen_classes.json", {"dataset": key, "fold": "known75", "classes": sorted(unseen_set)})
    criteria_known = {x: x.replace("_", " ") for x in known}
    splits = text_examples(key)
    train_source = splits.get("train", [])
    train_rows = (record(f"train/{sid}", {"text": text}, choice_question(criteria_known), label,
                         key, "train_known", source_id=sid, fold="known75")
                  for sid, text, label in train_source if label in known_set)
    counts = {"train_known": write_jsonl(out / "train_known.jsonl", train_rows)}
    test_source = list(splits.get("test", []))
    if key == "clinc150_bolt":
        test_source.extend(splits.get("oos_test", []))
    closed_rows, unknown_rows, supplied_rows = [], [], []
    for sid, text, label in test_source:
        if label in known_set:
            closed_rows.append(record(f"closed/{sid}", {"text": text}, choice_question(criteria_known), label,
                                      key, "test", source_id=sid, fold="known75", protocol="OW-0-closed"))
            unknown_rows.append(record(f"unknown_absent/{sid}", {"text": text},
                                       choice_question({**criteria_known, "UNKNOWN": "The example does not belong to a known class."}),
                                       label, key, "test", source_id=sid, fold="known75", protocol="OW-1-unknown-absent"))
        elif label in unseen_set:
            unknown_rows.append(record(f"unknown_absent/{sid}", {"text": text},
                                       choice_question({**criteria_known, "UNKNOWN": "The example does not belong to a known class."}),
                                       "UNKNOWN", key, "test", source_id=sid, fold="known75", protocol="OW-1-unknown-absent", true_class=label))
            supplied_rows.append(record(f"unseen_supplied/{sid}", {"text": text},
                                        choice_question({**criteria_known, label: label.replace("_", " "),
                                                         "UNKNOWN": "The example does not belong to a known class."}),
                                        label, key, "test", source_id=sid, fold="known75", protocol="OW-2-runtime-supplied",
                                        supplied_class=label))
        elif key == "clinc150_bolt" and label == "oos":
            unknown_rows.append(record(f"unknown_absent/{sid}", {"text": text},
                                       choice_question({**criteria_known, "UNKNOWN": "The example does not belong to a known class."}),
                                       "UNKNOWN", key, "test", source_id=sid, fold="known75", protocol="OW-1-unknown-absent",
                                       native_clinc_oos=True))
    counts["closed"] = write_jsonl(out / "closed.jsonl", closed_rows)
    counts["unknown_absent"] = write_jsonl(out / "unknown_absent.jsonl", unknown_rows)
    counts["unseen_supplied"] = write_jsonl(out / "unseen_supplied.jsonl", supplied_rows)
    write_metadata(out / "protocol_summary.json", {"dataset": key, "fold": "known75", "classes": {
        "all": len(all_labels), "known": len(known_set), "unseen": len(unseen_set)}, "records": counts,
        "split_source": "BOLT known75 class partition applied to the original pinned raw dataset splits; BOLT's processed row sampling is not substituted.",
        "notes": ["OW-1 includes known and held-out-class examples; held-out targets are UNKNOWN.",
                  "Native CLINC OOS examples are marked separately from BOLT held-out intents." if key == "clinc150_bolt" else ""]})
    write_metadata(split_dir / "split_manifest.json", {"dataset": key, "fold": "known75", "known_classes": known,
        "unseen_classes": sorted(unseen_set), "raw_split_counts": {name: len(rows) for name, rows in splits.items()},
        "derived_counts": counts, "protocols": ["OW-0-closed", "OW-1-unknown-absent", "OW-2-runtime-supplied"],
        "source_note": "BOLT fold defines class membership; evaluation rows are derived from the referenced pinned raw source."})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("system_one", "openworld", "vision", "all"), default="all")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry.yaml").read_text())["datasets"]
    groups = {"system_one": [k for k,v in registry.items() if v["group"] == "system_one"],
              "openworld": [k for k,v in registry.items() if v["group"] == "openworld"],
              "vision": [k for k,v in registry.items() if v["group"] == "vision"]}
    if args.dataset:
        keys = [args.dataset]
    elif args.group == "all":
        keys = sum(groups.values(), [])
    else:
        keys = groups[args.group]
    for key in keys:
        counts = openworld(key) if key.endswith("_bolt") else system_one(key)
        status_path = ROOT / registry[key]["path"] / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update({"prepared": True, "validated": False, "prepared_counts": counts})
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"PREPARED {key}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
