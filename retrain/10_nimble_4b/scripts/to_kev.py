#!/usr/bin/env python3
"""Convert the frozen Nimble split to Kev's labelled request JSONL format."""
import argparse, json
from pathlib import Path

def convert(row):
    request = json.loads(json.dumps(row["input"], ensure_ascii=False))
    questions = request.get("questions", {})
    if list(questions) != ["decision"]:
        raise ValueError(f"expected one decision question in {row.get('id')}")
    questions["decision"]["label"] = row["reference"]["target"]
    return {"state": request["state"], "questions": questions,
            "_meta": {"source": "nimble_controlled", "id": row["id"], "family": row.get("family")}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    rows=[convert(json.loads(line)) for line in a.input.open(encoding="utf-8") if line.strip()]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"output":str(a.output)}))
if __name__ == "__main__": main()
