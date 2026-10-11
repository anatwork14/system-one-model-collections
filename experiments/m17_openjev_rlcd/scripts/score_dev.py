#!/usr/bin/env python3
"""Fixed, dev-only raw proper-score check for cautious LoRA LR selection."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_m17 import ADAPTER_NAME, RationaleReader, load_adapter_state, load_policy, seed_everything
from openjev_rlcd.tasks import TASKS
from openjev_rlcd.evaluation import score


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cap", type=int, default=64)
    ap.add_argument("--seed", type=int, default=88017)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("dev LR selection requires the VKong GPU")
    seed_everything(args.seed)
    tok, model, _ = load_policy()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_adapter_state(model, state["adapter"], ADAPTER_NAME)
    model.set_adapter(ADAPTER_NAME, inference_mode=True)
    model.eval()
    reader = RationaleReader(tok)
    dev = TASKS["gsm8kv"]["load"]()["dev"][:args.cap]
    probs = []
    for j, item in enumerate(dev):
        random.seed(f"{args.seed}-{item['uid']}")
        torch.manual_seed(args.seed + j)
        group = reader.sample(model, [item], 1, 320, micro=1)[0]
        u, _ = reader.forward(model, group, grad=False, need_logp=False)
        probs.append(u[0].float().cpu().tolist())
    metrics = score(dev, probs)
    result = {"checkpoint": str(args.checkpoint), "split": "dev", "n": len(dev),
              "uids": [x["uid"] for x in dev], "selection_metric": "raw Brier (lower is better)",
              "metrics": metrics, "temperature_scaling": False}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"n": len(dev), "brier": metrics["brier"], "nll": metrics["nll"],
                      "acc": metrics["acc"], "output": str(out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
