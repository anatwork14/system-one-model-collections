#!/usr/bin/env python3
"""Score Kev-format records with a frozen Qwen backbone and local head-only checkpoint."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, default=Path("upstream"))
    p.add_argument("--checkpoint", type=Path, default=Path("weights/head_only"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    sys.path.insert(0, str(a.upstream.resolve()))
    from kev.data import load_records, materialize
    from kev.model import DecisionModel, load_tokenizer, training_context
    cfg = json.loads((a.checkpoint / "config.json").read_text(encoding="utf-8"))
    device = torch.device(a.device)
    tok = load_tokenizer(cfg["base"], revision=cfg["base_revision"])
    model = DecisionModel(cfg["base"], tok, a.device, revision=cfg["base_revision"],
                          head_dim=cfg["head_dim"], dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.head.load_state_dict(torch.load(a.checkpoint / "pointer_head.pt", map_location=a.device, weights_only=True))
    model.eval()
    ctx = training_context()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as dst:
        for request in load_records(a.input):
            start = time.perf_counter()
            rec = materialize(request)
            enc = model.encode(tok, rec, strict=True, max_state=ctx["max_state"], max_branch=ctx["max_branch"])
            with torch.inference_mode():
                logits = model(enc)
                scores = [z.float() for z in logits]
                probs = [z.softmax(-1).cpu().tolist() for z in scores]
            answer = {}
            predicted = {}
            labels = {}
            for q, pvec in zip(rec["questions"], probs):
                mapping = {key: float(prob) for key, prob in zip(q["keys"], pvec)}
                answer[q["qid"]] = mapping
                predicted[q["qid"]] = max(mapping, key=mapping.get)
                labels[q["qid"]] = q["keys"][q["label"]]
            dst.write(json.dumps({"id": request.get("_meta", {}).get("id"),
                                  "label": next(iter(labels.values())) if len(labels) == 1 else labels,
                                  "prediction": next(iter(predicted.values())) if len(predicted) == 1 else predicted,
                                  "probabilities": next(iter(answer.values())) if len(answer) == 1 else answer,
                                  "latency_ms": (time.perf_counter() - start) * 1000}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
