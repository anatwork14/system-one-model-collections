#!/usr/bin/env python3
"""Evaluate base, Stage 1 and Stage 2 on identical held-out UID sets."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_m17 import (ADAPTER_NAME, MODEL_ID, MODEL_REVISION, RationaleReader,
                       adapter_state, load_adapter_state, load_policy)
from openjev_rlcd.tasks import TASKS
from openjev_rlcd import evaluation as upstream_eval


@torch.no_grad()
def collect(model, reader, items, m: int, cap_batch: int = 8):
    single, mixture, forced, lengths, repetitions = [], [], [], [], []
    for start in range(0, len(items), cap_batch):
        batch = items[start:start + cap_batch]
        groups = reader.sample(model, batch, m, 320, micro=8)
        for item, group in zip(batch, groups):
            u, _ = reader.forward(model, group, grad=False, need_logp=False)
            probabilities = u.float().cpu().tolist()
            for (_, rationale, was_forced), probs in zip(group, probabilities):
                text = reader.tok.decode(rationale, skip_special_tokens=True)
                words = text.split()
                lengths.append(len(rationale))
                repetitions.append(1.0 - len(set(words)) / max(1, len(words)))
                forced.append(int(was_forced))
            single.append(probabilities[0])
            mixture.append([sum(p[k] for p in probabilities) / len(probabilities)
                            for k in range(len(probabilities[0]))])
    return {"single": single, "mixture": mixture,
            "rationale": {"mean_tokens": statistics.mean(lengths) if lengths else 0,
                          "p50_tokens": statistics.median(lengths) if lengths else 0,
                          "forced_marker_rate": statistics.mean(forced) if forced else 0,
                          "length_limit_rate": sum(n >= 320 for n in lengths) / max(1, len(lengths)),
                          "repetition_ratio": statistics.mean(repetitions) if repetitions else 0}}


def eval_arm(name, model, reader, dev, test, out: Path, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if name == "base":
        ctx = model.disable_adapter()
    else:
        ctx = None
    if ctx is None:
        dev_out = collect(model, reader, dev, 4)
        test_out = collect(model, reader, test, 4)
    else:
        with ctx:
            dev_out = collect(model, reader, dev, 4)
            test_out = collect(model, reader, test, 4)
    upstream_eval.SPLITS = {"dev": dev, "test": test}
    result = {"arm": name, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "dev_uids": [x["uid"] for x in dev], "test_uids": [x["uid"] for x in test],
              "dev_labels": [x["q"].index(1.0) for x in dev],
              "test_labels": [x["q"].index(1.0) for x in test],
              "primary_readout": "single", "secondary_readout": "mixture",
              "rationale_stats": {"dev": dev_out["rationale"], "test": test_out["rationale"]},
              "readouts": {}}
    for readout in ("single", "mixture"):
        raw, scaled, temperature = upstream_eval.scored(test_out[readout], dev_out[readout], seed)
        result["readouts"][readout] = {"raw": raw, "temperature": temperature,
                                       "temperature_scaled": scaled,
                                       "temperature_fit_split": "dev",
                                       "temperature_fit_n": len(dev)}
    latency = []
    latency_items = dev[:33]
    for j, item in enumerate(latency_items):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        groups = reader.sample(model, [item], 1, 320, micro=1)
        reader.forward(model, groups[0], grad=False, need_logp=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if j >= 3:
            latency.append(elapsed)
    ordered = sorted(latency)
    result["single_query_latency_seconds"] = {
        "n": len(ordered), "p50": ordered[len(ordered) // 2] if ordered else None,
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None,
        "note": "includes one rationale generation and answer-readout forward; first three are warm-up",
    }
    # Per-item distributions permit paired comparisons and independent recalculation.
    result["predictions"] = {"dev_single": dev_out["single"], "dev_mixture": dev_out["mixture"],
                             "test_single": test_out["single"], "test_mixture": test_out["mixture"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dev-cap", type=int, default=300)
    ap.add_argument("--test-cap", type=int, default=1260)
    ap.add_argument("--seed", type=int, default=44017)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M17 evaluation requires VKong CUDA; no local-GPU fallback is permitted")
    tok, model, _ = load_policy()
    reader = RationaleReader(tok)
    splits = TASKS["gsm8kv"]["load"]()
    dev, test = splits["dev"][:args.dev_cap], splits["test"][:args.test_cap]
    out = Path(args.output_dir)
    arms = []
    start = time.perf_counter()
    base = eval_arm("base", model, reader, dev, test, out / "base.json", args.seed)
    arms.append(base)
    for name, checkpoint in (("stage1", args.stage1), ("stage2", args.stage2)):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        load_adapter_state(model, state["adapter"], ADAPTER_NAME)
        model.set_adapter(ADAPTER_NAME)
        model.eval()
        arms.append(eval_arm(name, model, reader, dev, test, out / f"{name}.json", args.seed))
    summary = {"model_revision": MODEL_REVISION, "test_uids": [x["uid"] for x in test],
               "dev_uids": [x["uid"] for x in dev], "same_test_uid_order": True,
               "arms": {a["arm"]: a["readouts"]["single"] for a in arms},
               "evaluation_wall_seconds": time.perf_counter() - start,
               "calibration_note": "Temperature fitted on the same fixed dev pool for each arm; no test-driven selection."}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
