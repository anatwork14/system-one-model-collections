#!/usr/bin/env python3
"""Qwen3.5/LoRA adapter for the pinned OpenJev-RLCD objective."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parents[1]
UPSTREAM = HERE / "upstream"
import sys
sys.path.insert(0, str(UPSTREAM))
from openjev_rlcd.estimators import lambda_surrogate  # noqa: E402
from openjev_rlcd.tasks import TASKS, prompt  # noqa: E402
from rlcd_objective import rlcd_loss  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97"
ADAPTER_NAME = "policy"
REFERENCE_NAME = "stage1_reference"
MARKER = "Answer:"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> random.Random:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def discover_lora_targets(model: torch.nn.Module) -> list[str]:
    """Find Qwen3.5 text projection module types from the instantiated model."""
    wanted = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    }
    found = sorted({name.rsplit(".", 1)[-1] for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in wanted})
    if not {"in_proj_qkv", "q_proj"}.issubset(found):
        raise RuntimeError(f"Qwen3.5 hybrid text projections were not discovered: {found}")
    return found


def load_policy(model_id: str = MODEL_ID, revision: str = MODEL_REVISION, rank: int = 16,
                alpha: int = 32, dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    text_config = Qwen3_5TextConfig.from_pretrained(model_id, revision=revision)
    base = Qwen3_5ForCausalLM.from_pretrained(
        model_id, revision=revision, config=text_config, dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False,
    )
    targets = discover_lora_targets(base)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora_cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                          target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora_cfg, adapter_name=ADAPTER_NAME)
    model.config.use_cache = False
    return tok, model, targets


class RationaleReader:
    def __init__(self, tok):
        self.tok = tok
        ids = [tok.encode(" " + label, add_special_tokens=False) for label in TASKS["gsm8kv"]["labels"]]
        if any(len(x) != 1 for x in ids):
            raise ValueError(f"yes/no labels must each be exactly one token; got {ids}")
        self.label_ids = [x[0] for x in ids]
        if len(set(self.label_ids)) != len(self.label_ids):
            raise ValueError(f"yes/no label IDs must be unique; got {self.label_ids}")
        self.eos = {x for x in (tok.eos_token_id, tok.pad_token_id) if x is not None}
        self.suffix = tok.encode("\n" + MARKER, add_special_tokens=False)
        if not self.suffix:
            raise ValueError("Answer marker tokenization is empty")

    @torch.no_grad()
    def sample(self, model, items, m: int, max_new_tokens: int, micro: int = 8):
        model.eval()
        requests = [(i, prompt(self.tok, item, "gsm8kv")) for i, item in enumerate(items) for _ in range(m)]
        groups: list[list[tuple[list[int], list[int], bool]]] = [[] for _ in items]
        for start in range(0, len(requests), micro):
            chunk = requests[start:start + micro]
            enc = self.tok([p for _, p in chunk], return_tensors="pt", padding=True,
                           padding_side="left").to("cuda")
            generated = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=max_new_tokens, stop_strings=[MARKER], tokenizer=self.tok,
                pad_token_id=self.tok.pad_token_id, use_cache=True,
            )
            width = enc["input_ids"].shape[1]
            for (idx, _), row, prow in zip(chunk, generated, enc["input_ids"]):
                completion = row[width:].tolist()
                eos_at = next((j for j, token in enumerate(completion) if token in self.eos), None)
                head = completion if eos_at is None else completion[:eos_at]
                if self.tok.decode(head, skip_special_tokens=True).rstrip().endswith(MARKER):
                    rationale, forced = head, False
                else:
                    rationale = completion if eos_at is None else completion[:eos_at + 1]
                    forced = True
                prompt_ids = prow[prow != self.tok.pad_token_id].tolist()
                groups[idx].append((prompt_ids, rationale, forced))
        return groups

    def forward(self, model, group, grad: bool, need_logp: bool | None = None, chunk: int = 512):
        need_logp = grad if need_logp is None else need_logp
        seqs, spans = [], []
        for p, rationale, forced in group:
            context = p + rationale + (self.suffix if forced else [])
            seqs.append(context)
            spans.append((len(p), len(p) + len(rationale)))
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        positions = torch.arange(len(seqs), device="cuda")
        last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        base = model.get_base_model()
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                return_dict=True).last_hidden_state
            z = base.lm_head(hidden[positions, last])[:, self.label_ids].float()
            u = z.softmax(-1)
            if not need_logp:
                return u, None
            pos = torch.arange(width - 1, device="cuda")[None, :]
            start = torch.tensor([a for a, _ in spans], device="cuda")[:, None]
            end = torch.tensor([b for _, b in spans], device="cuda")[:, None]
            batch_idx, token_idx = ((pos >= start - 1) & (pos < end - 1)).nonzero(as_tuple=True)
            targets = ids[batch_idx, token_idx + 1]
            pieces = []
            for s in range(0, len(batch_idx), chunk):
                hs = hidden[batch_idx[s:s + chunk], token_idx[s:s + chunk]]
                if grad:
                    pieces.append(checkpoint(_token_logp, base.lm_head, hs, targets[s:s + chunk],
                                             use_reentrant=False))
                else:
                    pieces.append(_token_logp(base.lm_head, hs, targets[s:s + chunk]))
            logps = torch.cat(pieces) if pieces else torch.zeros(0, device="cuda")
        logp_r = torch.zeros(len(seqs), device="cuda").index_add(0, batch_idx, logps)
        return u, logp_r


def _token_logp(head, hidden, target):
    return head(hidden).float().log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)


def atomic_save(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(state, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def adapter_state(model, name: str = ADAPTER_NAME):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model, adapter_name=name).items()}


def load_adapter_state(model, state, name: str = ADAPTER_NAME):
    from peft import set_peft_model_state_dict
    result = set_peft_model_state_dict(model, state, adapter_name=name)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected adapter checkpoint keys: {result.unexpected_keys}")


def checkpoint_state(model, optimizer, step: int, seed: int, elapsed_gpu: float, config: dict,
                     reference_sha: str | None = None, sampler_rng=None, seen_uids=None):
    return {
        "format": "m17-adapter-resume-v1", "step": step, "seed": seed,
        "elapsed_gpu_seconds": elapsed_gpu, "scheduler": None, "config": config,
        "adapter": adapter_state(model), "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "reference_checkpoint_sha256": reference_sha,
        "sampler_rng": sampler_rng.getstate() if sampler_rng is not None else None,
        "seen_train_uids": sorted(seen_uids or ()),
    }


def validate_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"non-finite {name}")


def elapsed_training_seconds(previous_seconds: float, segment_started: float,
                             now: float | None = None) -> float:
    """Return stage-cumulative elapsed time across exact checkpoint resumes."""
    current = time.perf_counter() if now is None else now
    return previous_seconds + max(0.0, current - segment_started)


def gpu_telemetry():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5,
        )
        values = [int(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
        return dict(utilization_pct=values[0], memory_utilization_pct=values[1],
                    memory_used_mib=values[2], memory_total_mib=values[3])
    except Exception:
        return None


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("M17 training requires the VKong CUDA GPU; no local-GPU fallback is permitted")
    if args.stage not in {"stage1", "stage2"}:
        raise ValueError(args.stage)
    rng = seed_everything(args.seed)
    seen: set[str] = set()
    tok, model, targets = load_policy(rank=args.lora_rank, alpha=args.lora_alpha)
    reader = RationaleReader(tok)
    model.print_trainable_parameters()
    if args.stage == "stage2":
        if not args.init_checkpoint:
            raise ValueError("Stage 2 requires --init-checkpoint from Stage 1")
        ref = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_adapter_state(model, ref["adapter"], ADAPTER_NAME)
        from copy import deepcopy
        ref_config = deepcopy(model.peft_config[ADAPTER_NAME])
        ref_config.inference_mode = True
        model.add_adapter(REFERENCE_NAME, ref_config)
        load_adapter_state(model, ref["adapter"], REFERENCE_NAME)
        model.set_adapter(REFERENCE_NAME, inference_mode=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.set_adapter(ADAPTER_NAME, inference_mode=False)
        # Only the policy adapter may be optimized; the reference copy is frozen.
        for n, p in model.named_parameters():
            p.requires_grad_("." + ADAPTER_NAME + "." in n)
        ref_sha = sha256(Path(args.init_checkpoint))
    else:
        ref_sha = None

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable policy LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    config = vars(args).copy()
    config.update({"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "lora_targets": targets,
                   "label_ids": reader.label_ids, "upstream_commit": "22db40ad3333f5cf0bc5139097da70e0f8c05018"})
    previous_elapsed_gpu = 0.0
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed.get("format") != "m17-adapter-resume-v1":
            raise ValueError("incompatible checkpoint format")
        load_adapter_state(model, resumed["adapter"])
        optimizer.load_state_dict(resumed["optimizer"])
        random.setstate(resumed["python_rng"])
        np.random.set_state(resumed["numpy_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        if resumed.get("sampler_rng") is not None:
            rng.setstate(resumed["sampler_rng"])
        seen.update(resumed.get("seen_train_uids", []))
        args.start_step = int(resumed["step"])
        previous_elapsed_gpu = float(resumed.get("elapsed_gpu_seconds", 0.0))

    data_path = UPSTREAM / "data/gsm8k_verify.json"
    dataset_sha = sha256(data_path)
    if dataset_sha != "75d1cb9b69a5f2a1a84b9b003497e766217e73836e7808f088d6c0b3d32c467a":
        raise RuntimeError("released GSM8K-Verify data checksum changed")
    splits = TASKS["gsm8kv"]["load"]()
    if len(splits["train"]) != 4000 or len(splits["dev"]) != 500 or len(splits["test"]) != 1260:
        raise RuntimeError(f"unexpected split sizes: { {k: len(v) for k,v in splits.items()} }")
    config.update({
        "dataset_sha256": dataset_sha,
        "split_seed": 0,
        "split_uid_sha256": {name: hashlib.sha256("\n".join(sorted(x["uid"] for x in splits[name])).encode()).hexdigest()
                             for name in ("train", "dev", "test")},
        "dev_cap": 300, "test_cap": 1260, "m_eval": 4,
        "temperature_scaling_fit_split": "dev-only",
        "temperature_scaling_and_selection_share_dev": True,
    })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"
    if args.start_step == 0:
        log_path.write_text("")
    else:
        with log_path.open("a") as _:
            pass
    start_gpu = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    completed = args.start_step
    for step in range(args.start_step + 1, args.steps + 1):
        if args.max_gpu_seconds and elapsed_training_seconds(previous_elapsed_gpu, start_gpu) >= args.max_gpu_seconds:
            break
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        items = [rng.choice(splits["train"]) for _ in range(args.items_per_step)]
        seen.update(it["uid"] for it in items)
        groups = reader.sample(model, items, args.m, args.max_new_tokens, micro=args.generation_microbatch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = step_score = step_kl = forced = logp_tokens = limited = 0.0
        for item, group in zip(items, groups):
            label = 0 if item["correct"] else 1
            u, logp_r = reader.forward(model, group, grad=True)
            validate_finite("answer probabilities", u)
            validate_finite("rationale log probabilities", logp_r)
            k = len(TASKS["gsm8kv"]["labels"])
            ref_lp = None
            if args.stage == "stage2":
                with torch.no_grad():
                    model.set_adapter(REFERENCE_NAME, inference_mode=True)
                    _, ref_lp = reader.forward(model, group, grad=False, need_logp=True)
                    model.set_adapter(ADAPTER_NAME, inference_mode=False)
                validate_finite("frozen-reference rationale log probabilities", ref_lp)
                ratio = logp_r.detach() - ref_lp
                validate_finite("policy/reference log-ratio", ratio)
                step_kl += ratio.mean().item()
            loss = rlcd_loss(u, logp_r, label, stage=args.stage, sf_coef=args.sf_coef,
                             kl_beta=args.kl_beta, reference_logp=ref_lp,
                             diversity_weight=args.diversity_weight)
            (loss / len(items)).backward()
            step_loss += loss.detach().item() / len(items)
            step_score += u.detach()[:, label].mean().item() / len(items)
            forced += sum(bool(f) for _, _, f in group)
            logp_tokens += sum(len(r) for _, r, _ in group)
            limited += sum(len(r) >= args.max_new_tokens for _, r, _ in group)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        validate_finite("gradient norm", torch.as_tensor(grad_norm))
        if float(grad_norm) <= 0.0:
            raise FloatingPointError("zero gradient norm")
        if args.stage == "stage2" and abs(step_kl / len(items)) > 25.0:
            raise FloatingPointError("policy/reference rationale KL ratio exceeded the safety threshold")
        if forced / (len(items) * args.m) > 0.98:
            raise FloatingPointError("nearly every rationale missed the Answer: stop marker")
        optimizer.step()
        torch.cuda.synchronize()
        now = elapsed_training_seconds(previous_elapsed_gpu, start_gpu)
        step_seconds = time.perf_counter() - step_started
        completed = step
        record = {"step": step, "loss": step_loss, "mean_correct_label_probability": step_score,
                  "mean_log_ratio": step_kl / len(items) if args.stage == "stage2" else 0.0,
                  "grad_norm": float(grad_norm), "forced_marker_rate": forced / (len(items) * args.m),
                  "rationale_tokens": int(logp_tokens), "step_seconds": step_seconds,
                  "mean_rationale_tokens": logp_tokens / (len(items) * args.m),
                  "stage_elapsed_seconds": now,
                  "length_limit_rate": limited / (len(items) * args.m),
                  "unique_train_uids_seen": len(seen), "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved()}
        telemetry = gpu_telemetry() if step % args.checkpoint_every == 0 or step == args.steps else None
        if telemetry:
            record["gpu_telemetry"] = telemetry
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            state = checkpoint_state(model, optimizer, step, args.seed, now, config, ref_sha,
                                     sampler_rng=rng, seen_uids=seen)
            atomic_save(out / "checkpoint.pt", state)
    # Always checkpoint the last completed step, including an early time-limit stop.
    last = completed
    saved = -1
    if (out / "checkpoint.pt").exists():
        try:
            saved = int(torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)["step"])
        except Exception:
            saved = -1
    if last != saved:
        state = checkpoint_state(model, optimizer, last, args.seed,
                                 elapsed_training_seconds(previous_elapsed_gpu, start_gpu), config, ref_sha,
                                 sampler_rng=rng, seen_uids=seen)
        atomic_save(out / "checkpoint.pt", state)
    print(json.dumps({"stage": args.stage, "completed_steps": last, "target_steps": args.steps,
                      "gpu_seconds": time.perf_counter() - start_gpu,
                      "seen_unique_train_uids": len(seen), "checkpoint": str(out / "checkpoint.pt")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--items-per-step", type=int, default=8)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--diversity-weight", type=float, default=0.0)
    ap.add_argument("--sf-coef", type=float, default=0.3)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--generation-microbatch", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--max-gpu-seconds", type=float, default=0.0)
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--output-dir", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
