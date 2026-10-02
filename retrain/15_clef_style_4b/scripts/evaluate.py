#!/usr/bin/env python3
"""Score saved per-question logits against a canonical split."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def softmax(logits, temperature=1.0):
    values = [float(value) / temperature for value in logits]
    peak = max(values)
    exp = [math.exp(value - peak) for value in values]
    total = sum(exp)
    return [value / total for value in exp]


def target_id(question):
    target = question["target"]
    if question["type"] == "noul":
        return "true" if target else "false"
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL: record_id, question_id, option_ids, logits")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = {row["id"]: row for row in (json.loads(line) for line in args.data.read_text().splitlines() if line.strip())}
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    per_question = []
    record_correct = defaultdict(list)
    expected = {(record_id, question_id) for record_id, record in records.items() for question_id in record["questions"]}
    seen = set()
    for row in predictions:
        record_id, question_id = row["record_id"], row["question_id"]
        pair = (record_id, question_id)
        if pair in seen:
            raise ValueError(f"duplicate prediction for {record_id}/{question_id}")
        seen.add(pair)
        question = records[record_id]["questions"][question_id]
        option_ids = [str(value) for value in row["option_ids"]]
        if len(option_ids) != len(row["logits"]) or target_id(question) not in option_ids:
            raise ValueError(f"option mapping invalid for {record_id}/{question_id}")
        probs = softmax(row["logits"], args.temperature)
        target = option_ids.index(target_id(question))
        predicted = max(range(len(probs)), key=probs.__getitem__)
        brier = sum((prob - (index == target)) ** 2 for index, prob in enumerate(probs))
        per_question.append({"correct": predicted == target, "nll": -math.log(max(probs[target], 1e-12)), "brier": brier, "confidence": probs[predicted], "risk": predicted != target})
        record_correct[record_id].append(predicted == target)
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(f"prediction coverage mismatch: {len(missing)} missing, {len(extra)} extra")
    bins = [[] for _ in range(15)]
    for item in per_question:
        bins[min(14, int(item["confidence"] * 15))].append(item)
    ece = sum(len(values) / max(1, len(per_question)) * abs(sum(x["confidence"] for x in values) / len(values) - sum(x["correct"] for x in values) / len(values)) for values in bins if values)
    ordered = sorted(per_question, key=lambda item: item["confidence"], reverse=True)
    aurc = sum(sum(item["risk"] for item in ordered[:end]) / end for end in range(1, len(ordered) + 1)) / max(1, len(ordered))
    report = {
        "questions": len(per_question),
        "records": len(record_correct),
        "accuracy": sum(x["correct"] for x in per_question) / max(1, len(per_question)),
        "nll": sum(x["nll"] for x in per_question) / max(1, len(per_question)),
        "brier": sum(x["brier"] for x in per_question) / max(1, len(per_question)),
        "ece_15": ece,
        "aurc": aurc,
        "case_exact_accuracy": sum(all(values) for values in record_correct.values()) / max(1, len(record_correct)),
        "temperature": args.temperature,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
