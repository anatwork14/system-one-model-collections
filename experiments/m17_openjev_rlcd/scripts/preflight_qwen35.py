#!/usr/bin/env python3
"""GPU gate: load Qwen3.5 text stack, inspect 20 prompts, and backprop through LoRA."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_m17 import (MODEL_ID, MODEL_REVISION, RationaleReader, load_policy,
                       seed_everything)
from rlcd_objective import rlcd_loss
from openjev_rlcd.tasks import TASKS, prompt
from transformers import Qwen3_5ForConditionalGeneration


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("architecture preflight is only valid on the VKong GPU")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok, model, targets = load_policy()
    reader = RationaleReader(tok)
    # Verify text-only weights against the official multimodal class on identical text.
    first = TASKS["gsm8kv"]["load"]()["dev"][0]
    encoded = tok(prompt(tok, first, "gsm8kv"), return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_logits = model.get_base_model()(**encoded, use_cache=False).logits[0, -1].float()
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False,
    ).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full_logits = full_model(**encoded, use_cache=False).logits[0, -1].float()
    max_abs = (text_logits - full_logits).abs().max().item()
    if not torch.allclose(text_logits, full_logits, rtol=2e-2, atol=2e-2):
        raise AssertionError(f"text-only logits differ from official conditional model: {max_abs}")
    del full_model
    torch.cuda.empty_cache()
    labels = [tok.encode(" " + x, add_special_tokens=False) for x in TASKS["gsm8kv"]["labels"]]
    if len(labels) != 2 or any(len(x) != 1 for x in labels) or labels[0][0] == labels[1][0]:
        raise AssertionError(f"invalid label tokenization: {labels}")
    splits = TASKS["gsm8kv"]["load"]()
    examples = []
    for j, item in enumerate(splits["dev"][:20]):
        seed_everything(117 + j)
        group = reader.sample(model, [item], 1, 320, micro=1)[0]
        u, _ = reader.forward(model, group, grad=False, need_logp=False)
        probs = u[0].float().cpu().tolist()
        if not all(math.isfinite(p) for p in probs) or abs(sum(probs) - 1.0) > 1e-5:
            raise AssertionError(f"invalid candidate probabilities for {item['uid']}: {probs}")
        rationale_text = tok.decode(group[0][1], skip_special_tokens=True)
        if not group[0][2] and not rationale_text.rstrip().endswith("Answer:"):
            raise AssertionError("generation stopped without the emitted Answer: boundary")
        if "<think>" in rationale_text or "</think>" in rationale_text:
            raise AssertionError("native thinking block leaked into explicit rationale")
        examples.append({"uid": item["uid"], "marker_forced": group[0][2],
                         "rationale_tokens": len(group[0][1]), "answer_probs": probs,
                         "rationale_excerpt": rationale_text[:400]})
    if len(examples) != 20:
        raise AssertionError("exactly 20 generated examples are required for the prompt audit")
    if not any(not row["marker_forced"] for row in examples):
        raise AssertionError("none of the 20 rationales emitted a natural Answer: stop marker")

    # Exercise the gradient-checkpointed hybrid attention/DeltaNet stack end-to-end.
    model.train()
    seed_everything(178)
    group = reader.sample(model, splits["train"][:1], 2, 96, micro=2)[0]
    u, logp = reader.forward(model, group, grad=True)
    loss = rlcd_loss(u, logp, 0, stage="stage2", sf_coef=0.3, kl_beta=0.0,
                     reference_logp=logp.detach())
    loss.backward()
    grads = [(name, p.grad) for name, p in model.named_parameters()
             if "lora_" in name and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for _, g in grads):
        raise AssertionError("hybrid model produced missing or non-finite LoRA gradients")
    grad_norm = torch.linalg.vector_norm(torch.stack([g.float().norm() for _, g in grads])).item()
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise AssertionError(f"invalid LoRA gradient norm: {grad_norm}")
    torch.cuda.synchronize()
    report = {
        "status": "pass", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "loaded_class": type(model.get_base_model()).__name__, "config_class": type(model.get_base_model().config).__name__,
        "model_dtype": str(next(model.get_base_model().parameters()).dtype),
        "tokenizer_class": type(tok).__name__, "label_token_ids": [x[0] for x in labels],
        "label_probabilities_sum_checked": True, "lora_targets_discovered": targets,
        "text_only_vs_full_model_logits_max_abs_diff": max_abs,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "lora_gradient_norm": grad_norm, "lora_gradient_tensors": len(grads),
        "gpu": torch.cuda.get_device_name(0), "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime_versions": {x: version(x) for x in ("torch", "transformers", "peft", "accelerate", "safetensors")},
        "python": platform.python_version(), "elapsed_seconds": time.perf_counter() - t0,
        "prompt_audit_count": len(examples), "prompt_audit": examples,
    }
    out = Path(os.environ.get("M17_RUN_DIR", "/data/checkpoints/m17_openjev_rlcd")) / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
