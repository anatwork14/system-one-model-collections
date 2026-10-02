#!/usr/bin/env python3
"""Generate train-only question and candidate-order variants."""

import argparse
import json
import random
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--variants-per-record", type=int, default=2)
    args = parser.parse_args()
    randomizer = random.Random(args.seed)
    source = DATA / "train/nimble.jsonl"
    output = DATA / "augmented/train_schema_variants.jsonl"
    rows = []
    for line in source.read_text().splitlines():
        base = json.loads(line)
        for variant in range(args.variants_per_record):
            item = json.loads(json.dumps(base))
            questions = list(item["questions"].items())
            original_order = [key for key, _ in questions]
            for _ in range(8):
                randomizer.shuffle(questions)
                if [key for key, _ in questions] != original_order:
                    break
            if [key for key, _ in questions] == original_order and len(questions) > 1:
                questions = questions[1:] + questions[:1]
            item["questions"] = dict(questions)
            item["id"] = f"{base['id']}#schema-{variant + 1}"
            item["source_id"] = base["id"]
            item["augmentation"] = "question_order_permutation"
            rows.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for row in rows:
            # Preserve insertion order for the top-level questions mapping;
            # the official encoder consumes fields in that order.
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} train-only variants to {output}")


if __name__ == "__main__":
    main()
