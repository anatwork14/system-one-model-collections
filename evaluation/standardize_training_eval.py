#!/usr/bin/env python3
"""Normalize method-native inner-dev outputs for the shared metric calculator.

The output intentionally contains only fields the native artifacts support.
It is a metric input, not a complete serving trace; token, cost, and compute
measurements are reported unavailable by the experiment launcher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(rows, output: Path):
    if not rows:
        raise ValueError("normalization produced no evaluation rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def key_for_label(label, probabilities):
    if isinstance(label, bool):
        candidates = [str(label).lower(), str(label)]
    else:
        candidates = [str(label)]
    for candidate in candidates:
        if candidate in probabilities:
            return candidate
    raise ValueError(f"gold label {label!r} does not match probability keys {list(probabilities)}")


def normalize_m10(native: Path):
    report = json.loads(native.read_text(encoding="utf-8"))
    rows = []
    for row in report["rows"]:
        probabilities = {str(k).lower() if isinstance(k, bool) else str(k): float(v)
                         for k, v in row["probabilities"].items()}
        label = key_for_label(row["target"], probabilities)
        rows.append({"id": str(row["id"]), "label": label, "probabilities": probabilities})
    return rows


def normalize_m11(native: Path):
    rows = read_jsonl(native)
    output = []
    for row in rows:
        probabilities = {str(k): float(v) for k, v in row["probabilities"].items()}
        output.append({"id": str(row["id"]), "label": key_for_label(row["label"], probabilities),
                       "probabilities": probabilities})
    return output


def normalize_m12(native: Path, source: Path):
    predictions = {str(row["id"]): row for row in read_jsonl(native)}
    output = []
    for request in read_jsonl(source):
        request_id = str(request.get("_meta", {}).get("id", request.get("id", "")))
        prediction = predictions.get(request_id)
        if prediction is None:
            raise ValueError(f"missing M12 prediction for request {request_id}")
        labels = prediction["label"] if isinstance(prediction["label"], dict) else {
            next(iter(request["questions"])): prediction["label"]}
        distributions = prediction["probabilities"] if isinstance(prediction["probabilities"], dict) else {
            next(iter(request["questions"])): prediction["probabilities"]}
        for qid, question in request["questions"].items():
            probs = distributions[qid]
            if isinstance(probs, list):
                # The native infer adapter normally writes key-to-probability maps.
                raise ValueError(f"M12 probabilities for {request_id}/{qid} are not keyed by option")
            output.append({"id": f"{request_id}/{qid}", "label": key_for_label(labels[qid], probs),
                           "probabilities": {str(k): float(v) for k, v in probs.items()}})
    extra = set(predictions) - {str(r.get("_meta", {}).get("id", r.get("id", ""))) for r in read_jsonl(source)}
    if extra:
        raise ValueError(f"unexpected M12 prediction IDs: {sorted(extra)[:3]}")
    return output


def normalize_m13(native: Path):
    raw = json.loads(native.read_text(encoding="utf-8"))
    rows = []
    for row in raw:
        keys = [str(key) for key in row["keys"]]
        label_index = row["label"]
        if isinstance(label_index, bool) or not isinstance(label_index, int) or not 0 <= label_index < len(keys):
            raise ValueError(f"invalid M13 gold index for {row.get('id')}")
        probabilities = {key: float(value) for key, value in zip(keys, row["p"])}
        rows.append({"id": f"{row['id']}/{row.get('question', 'decision')}",
                     "label": keys[label_index], "probabilities": probabilities})
    return rows


def normalize_m14(native: Path):
    rows = []
    for row in read_jsonl(native):
        probabilities = {str(key): float(value) for key, value in row["probabilities"].items()}
        rows.append({"id": str(row["id"]), "label": key_for_label(row["target"], probabilities),
                     "probabilities": probabilities})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("M10", "M11", "M12", "M13", "M14"))
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="required for M12's multi-question request output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "M10":
        rows = normalize_m10(args.native)
    elif args.method == "M11":
        rows = normalize_m11(args.native)
    elif args.method == "M12":
        if not args.source:
            parser.error("--source is required for M12")
        rows = normalize_m12(args.native, args.source)
    elif args.method == "M13":
        rows = normalize_m13(args.native)
    else:
        rows = normalize_m14(args.native)
    write_rows(rows, args.output)
    print(f"normalized {len(rows)} evaluation decisions to {args.output}")


if __name__ == "__main__":
    main()
