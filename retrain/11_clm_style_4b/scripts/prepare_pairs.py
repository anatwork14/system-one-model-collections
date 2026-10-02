#!/usr/bin/env python3
"""Convert Nimble records into state/typed-candidate contrastive examples."""
import argparse
import json
from pathlib import Path

def state_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)

def convert(row):
    request = row["input"]
    if list(request["questions"]) != ["decision"]:
        raise ValueError(f"expected one decision question in {row.get('id')}")
    q = request["questions"]["decision"]
    kind, criteria = q["type"], q.get("criteria")
    if kind == "choice":
        candidates = [(str(k), str(v or k)) for k, v in criteria.items()]
        label = str(row["reference"]["target"])
    elif kind == "noul":
        desc = criteria or {}
        candidates = [("false", str(desc.get("false") or "The proposition is false.")),
                      ("true", str(desc.get("true") or "The proposition is true."))]
        label = str(bool(row["reference"]["target"])).lower()
    elif kind == "score":
        candidates = [(str(i), str(level)) for i, level in enumerate(criteria)]
        label = str(row["reference"]["target"])
    else:
        raise ValueError(f"unsupported type {kind!r}")
    keys = [key for key, _ in candidates]
    if label not in keys or len(keys) < 2:
        raise ValueError(f"invalid target/candidates in {row.get('id')}")
    return {
        "id": row["id"],
        "state": state_text(request["state"]),
        "instruction": str(q["instructions"]),
        "kind": kind,
        "candidates": [{"id": key, "text": text} for key, text in candidates],
        "label": label,
        "domain": row.get("domain"),
        "family": row.get("source_family", row.get("family")),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [convert(json.loads(line)) for line in a.input.open(encoding="utf-8") if line.strip()]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(a.output)}))

if __name__ == "__main__": main()
