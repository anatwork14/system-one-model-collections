#!/usr/bin/env python3
"""Score a saved candidate-logit adapter on the family-disjoint inner dev set."""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "upstream"))
from nimble.training.model_loading import load_base
from nimble.training.schema_data import prepare_data
from nimble.training.schema_train import CandidateCollator, evaluate


def main():
    adapter = HERE / "weights/nimble-4b-clean"
    contract_path = adapter / "schema_config.json"
    if not contract_path.is_file():
        raise SystemExit("Missing trained adapter contract: " + str(contract_path))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(adapter)
    _, holdout, _ = prepare_data(HERE / "data/inner_split", HERE / "data/inner_split/validation.jsonl", tok,
                                 contract["max_length"], contract["model"], contract["revision"])
    model = PeftModel.from_pretrained(load_base(contract["model"], contract["revision"]), adapter).eval()
    report = evaluate(model, holdout, CandidateCollator(tok.pad_token_id), HERE / "results/dev.json")
    (HERE / "results/dev_metrics.json").write_text(json.dumps(report["summary"], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
