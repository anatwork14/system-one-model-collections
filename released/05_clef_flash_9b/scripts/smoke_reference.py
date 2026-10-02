#!/usr/bin/env python3
"""Run one text-only three-field request through the local official 9B release."""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "weights" / "model"
sys.path.insert(0, str(MODEL))
from joint_schema_model import load_release_model, systemone  # noqa: E402


def main() -> None:
    if not (MODEL / "model.safetensors.index.json").exists():
        raise SystemExit(f"complete model weights are not present under {MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    dtype = torch.bfloat16
    model, processor = load_release_model(MODEL, device=device, dtype=dtype, low_cpu_mem_usage=True)
    request = {
        "model": "clef-flash",
        "state": {"ticket": "Checkout has failed for every customer since the last release."},
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments and invoices", "technical": "Outages and errors"}},
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "severity": {"type": "score", "criteria": ["No impact", "Minor", "Major", "Critical"]},
        },
    }
    response = systemone(model, processor, request, max_length=1024)
    result = {"device": device, "response": response}
    destination = ROOT / "results" / "official_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    answers = response["answers"]
    for key in request["questions"]:
        if key not in answers:
            raise AssertionError(f"missing answer for {key}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
