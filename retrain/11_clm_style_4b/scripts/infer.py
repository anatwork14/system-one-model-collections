#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import torch, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_contrastive import PairHeads, render

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, default=Path("weights/clm-heads"))
p.add_argument("--device", default="cuda")
a = p.parse_args()
cfg = json.loads((a.checkpoint / "config.json").read_text())
yaml.safe_load(a.config.read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(a.checkpoint)
model = AutoModelForCausalLM.from_pretrained(cfg["base"], dtype=torch.bfloat16, device_map=a.device).eval()
heads = PairHeads(cfg["hidden_size"], cfg["projection_dim"]).to(a.device)
heads.load_state_dict(torch.load(a.checkpoint / "heads.pt", map_location=a.device, weights_only=True)); heads.eval()
def vec(content):
    ids = render(tok, "user", content).to(a.device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
    return out.hidden_states[-1][0, -1].float()
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.input.open(encoding="utf-8") as src, a.output.open("w", encoding="utf-8") as dst:
    for line in src:
        row = json.loads(line); start = time.perf_counter()
        state = vec("State:\n" + row["state"] + "\n\nQuestion:\n" + row["instruction"])
        actions = torch.stack([vec("Candidate answer/action:\n" + c["id"] + ": " + c["text"]) for c in row["candidates"]])
        with torch.inference_mode():
            scores = heads.scores(state, actions)
            probs = scores.softmax(-1).cpu().tolist()
        mapping = {c["id"]: p for c, p in zip(row["candidates"], probs)}
        pred = max(mapping, key=mapping.get)
        dst.write(json.dumps({"id": row["id"], "label": row.get("label"), "prediction": pred,
                              "probabilities": mapping, "latency_ms": (time.perf_counter()-start)*1000}, ensure_ascii=False) + "\n")
