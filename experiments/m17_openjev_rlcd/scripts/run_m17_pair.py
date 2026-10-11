#!/usr/bin/env python3
"""Single-pair runner with smoke/timing gates and a hard 8-hour training ceiling."""
from __future__ import annotations

import json
import importlib.metadata
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("M17_RUN_DIR", "/data/checkpoints/m17_openjev_rlcd/run_" +
                          datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
PYTHON = sys.executable
LOG = OUT / "run.log"
TRAIN_BUDGET = 8 * 60 * 60
SAFETY_SECONDS = 10 * 60


def log(message: str) -> None:
    line = message.rstrip() + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
    sys.stdout.write(line)
    sys.stdout.flush()


def run(command: list[str], env: dict[str, str] | None = None) -> int:
    log("$ " + " ".join(command))
    proc = subprocess.Popen(command, cwd=ROOT, env=env or os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    assert proc.stdout is not None
    with LOG.open("ab") as logbin:
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            logbin.write(chunk)
            logbin.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    return proc.wait()


def train(stage: str, seed: int, directory: str, steps: int, items: int, m: int,
          max_tokens: int, init: Path | None = None, max_seconds: float = 0.0,
          lr: float = 2e-5) -> int:
    command = [PYTHON, "-u", "scripts/train_m17.py", "--stage", stage,
               "--seed", str(seed), "--steps", str(steps), "--items-per-step", str(items),
               "--m", str(m), "--max-new-tokens", str(max_tokens), "--lr", str(lr),
               "--checkpoint-every", "5" if steps <= 20 else "10",
               "--output-dir", str(OUT / directory)]
    if init:
        command += ["--init-checkpoint", str(init)]
    if max_seconds > 0:
        command += ["--max-gpu-seconds", str(max_seconds)]
    return run(command)


def training_records(directory: str):
    path = OUT / directory / "train.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gpu_step_times(directories: list[str]):
    result = {}
    for directory in directories:
        rows = training_records(directory)
        # Ignore the first two kernels/allocator warm-up steps when forecasting.
        times = [r["step_seconds"] for r in rows if r["step"] > 2]
        if not times:
            raise RuntimeError(f"no timed optimization steps recorded in {directory}")
        ordered = sorted(times)
        result[directory] = {"n": len(times), "mean": statistics.mean(times),
                             "p50": statistics.median(times),
                             "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]}
    return result


def training_seconds(directories: list[str]) -> float:
    return sum(sum(r["step_seconds"] for r in training_records(d)) for d in directories)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    (OUT / "environment.txt").write_text(subprocess.run([PYTHON, "-m", "pip", "freeze"],
                                                          check=True, capture_output=True, text=True).stdout)
    (OUT / "gpu.txt").write_text(subprocess.run(["nvidia-smi"], check=False,
                                                  capture_output=True, text=True).stdout)
    torch_mod = __import__("torch")
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "numpy", "safetensors", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    meminfo = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    environment = {
        "python": sys.version, "packages": versions,
        "cuda_version": torch_mod.version.cuda,
        "cuda_available": torch_mod.cuda.is_available(),
        "gpu_name": torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else None,
        "gpu_total_memory_bytes": torch_mod.cuda.get_device_properties(0).total_memory if torch_mod.cuda.is_available() else None,
        "host_memory_info": meminfo,
        "nvidia_smi": (OUT / "gpu.txt").read_text(),
        "upstream_commit": "22db40ad3333f5cf0bc5139097da70e0f8c05018",
        "model_revision": "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
    }
    serialized_environment = json.dumps(environment, indent=2) + "\n"
    (OUT / "environment.json").write_text(serialized_environment)
    (ROOT / "manifests/environment.json").write_text(serialized_environment)
    (OUT / "effective_config.json").write_text(json.dumps({
        "model": "Qwen/Qwen3.5-4B", "model_revision": "c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
        "train_items": 4000, "dev_items": 500, "test_items": 1260, "split_seed": 0,
        "target_steps_per_final_stage": 400, "pilot_steps_per_stage": 10,
        "smoke_steps_per_stage": 2, "items_per_step": 8, "m": 4, "max_new_tokens": 320,
        "diversity_weight": 0.0, "stage1_seed": 17, "stage2_seed": 1017,
        "stage2_sf_coef": 0.3, "stage2_kl_beta": 0.04, "training_gpu_budget_seconds": TRAIN_BUDGET,
        "safety_buffer_seconds": SAFETY_SECONDS,
    }, indent=2) + "\n")

    # CPU-only provenance tests are the first gate; no model is loaded until these pass.
    for command in ([PYTHON, "scripts/verify_upstream.py"],
                    [PYTHON, "-m", "pytest", "upstream/tests"],
                    [PYTHON, "-m", "pytest", "tests"],
                    [PYTHON, "scripts/audit_dataset.py"]):
        if run(list(command), env) != 0:
            raise RuntimeError(f"pre-training validation failed: {command}")
    if run([PYTHON, "scripts/preflight_qwen35.py"], env) != 0:
        raise RuntimeError("Qwen3.5 architecture/tokenizer/GPU preflight failed")

    # Small complete pass through both mathematical stages.
    if train("stage1", 17, "smoke/stage1", 2, 2, 2, 64) != 0:
        raise RuntimeError("Stage-1 smoke test failed")
    smoke1 = OUT / "smoke/stage1/checkpoint.pt"
    if train("stage2", 1017, "smoke/stage2", 2, 2, 2, 64, init=smoke1) != 0:
        raise RuntimeError("Stage-2 smoke test failed")

    # Cautious per-stage LoRA LR tuning uses a fixed dev-only Brier screen.
    candidate_lrs = [1e-5, 2e-5]
    s1_candidates = []
    for lr in candidate_lrs:
        tag = f"lr_tuning/stage1/lr_{lr:.0e}"
        if train("stage1", 17, tag, 5, 8, 4, 320, lr=lr) != 0:
            raise RuntimeError(f"Stage-1 LR pilot failed at {lr}")
        score_path = OUT / tag / "dev_score.json"
        if run([PYTHON, "scripts/score_dev.py", "--checkpoint", str(OUT / tag / "checkpoint.pt"),
                "--output", str(score_path), "--cap", "64", "--seed", "88017"], env) != 0:
            raise RuntimeError(f"Stage-1 dev-only LR scoring failed at {lr}")
        s1_candidates.append({"lr": lr, "score": json.loads(score_path.read_text())})
    best_s1 = min(s1_candidates, key=lambda x: (x["score"]["metrics"]["brier"], x["score"]["metrics"]["nll"]))
    s1_ref = OUT / f"lr_tuning/stage1/lr_{best_s1['lr']:.0e}/checkpoint.pt"

    s2_candidates = []
    for lr in candidate_lrs:
        tag = f"lr_tuning/stage2/lr_{lr:.0e}"
        if train("stage2", 1017, tag, 5, 8, 4, 320, init=s1_ref, lr=lr) != 0:
            raise RuntimeError(f"Stage-2 LR pilot failed at {lr}")
        score_path = OUT / tag / "dev_score.json"
        if run([PYTHON, "scripts/score_dev.py", "--checkpoint", str(OUT / tag / "checkpoint.pt"),
                "--output", str(score_path), "--cap", "64", "--seed", "88017"], env) != 0:
            raise RuntimeError(f"Stage-2 dev-only LR scoring failed at {lr}")
        s2_candidates.append({"lr": lr, "score": json.loads(score_path.read_text())})
    best_s2 = min(s2_candidates, key=lambda x: (x["score"]["metrics"]["brier"], x["score"]["metrics"]["nll"]))
    selection = {"criterion": "raw Brier on first 64 fixed dev UIDs; NLL breaks ties",
                 "selection_uses_test": False, "same_dev_pool_as_temperature_fit": True,
                 "stage1_candidates": [{"lr": x["lr"], "brier": x["score"]["metrics"]["brier"],
                                        "nll": x["score"]["metrics"]["nll"]} for x in s1_candidates],
                 "stage1_selected_lr": best_s1["lr"],
                 "stage2_candidates": [{"lr": x["lr"], "brier": x["score"]["metrics"]["brier"],
                                        "nll": x["score"]["metrics"]["nll"]} for x in s2_candidates],
                 "stage2_selected_lr": best_s2["lr"]}
    (OUT / "learning_rate_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    effective = json.loads((OUT / "effective_config.json").read_text())
    effective["stage1_lr"] = best_s1["lr"]
    effective["stage2_lr"] = best_s2["lr"]
    (OUT / "effective_config.json").write_text(json.dumps(effective, indent=2) + "\n")

    # Timed runs use the final item batch, rationale count, and 320-token cap.
    if train("stage1", 17, "pilot/stage1", 10, 8, 4, 320, lr=best_s1["lr"]) != 0:
        raise RuntimeError("Stage-1 timing pilot failed")
    pilot1 = OUT / "pilot/stage1/checkpoint.pt"
    if train("stage2", 1017, "pilot/stage2", 10, 8, 4, 320, init=pilot1, lr=best_s2["lr"]) != 0:
        raise RuntimeError("Stage-2 timing pilot failed")
    tuning = [f"lr_tuning/stage1/lr_{lr:.0e}" for lr in candidate_lrs]
    tuning += [f"lr_tuning/stage2/lr_{lr:.0e}" for lr in candidate_lrs]
    pilots = ["smoke/stage1", "smoke/stage2"] + tuning + ["pilot/stage1", "pilot/stage2"]
    pilot_seconds = training_seconds(pilots)
    timing = gpu_step_times(["pilot/stage1", "pilot/stage2"])
    p90_pair = timing["pilot/stage1"]["p90"] + timing["pilot/stage2"]["p90"]
    usable = max(0.0, TRAIN_BUDGET - pilot_seconds - SAFETY_SECONDS)
    target_steps = min(400, math.floor(usable / max(1.0, p90_pair * 1.15)))
    projection = {"pilot_training_seconds": pilot_seconds, "pilot_step_timing": timing,
                  "projected_final_steps_per_stage": target_steps,
                  "remaining_training_budget_seconds": usable,
                  "projection_margin": 1.15, "safety_buffer_seconds": SAFETY_SECONDS}
    (OUT / "throughput_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    log("TIMED_PROJECTION " + json.dumps(projection, sort_keys=True))
    if target_steps > 0:
        stage1_limit = target_steps * timing["pilot/stage1"]["p90"] * 1.15
        stage2_reserve = target_steps * timing["pilot/stage2"]["p90"] * 1.15
        if train("stage1", 17, "stage1", target_steps, 8, 4, 320,
                 max_seconds=stage1_limit, lr=best_s1["lr"]) != 0:
            raise RuntimeError("Stage-1 training failed")
        stage1_ckpt = OUT / "stage1/checkpoint.pt"
        elapsed = training_seconds(pilots + ["stage1"])
        global_left = max(0.0, TRAIN_BUDGET - elapsed - SAFETY_SECONDS)
        stage2_limit = min(stage2_reserve, global_left)
        if stage2_limit > 0 and train("stage2", 1017, "stage2", target_steps, 8, 4, 320,
                                      init=stage1_ckpt, max_seconds=stage2_limit,
                                      lr=best_s2["lr"]) != 0:
            raise RuntimeError("Stage-2 training failed")

    (OUT / "training_steps.json").write_text(json.dumps({
        "smoke": {"stage1": 2, "stage2": 2},
        "pilot": {"stage1": len(training_records("pilot/stage1")),
                  "stage2": len(training_records("pilot/stage2"))},
        "learning_rate_tuning_steps": sum(len(training_records(d)) for d in tuning),
        "final": {"stage1": len(training_records("stage1")),
                  "stage2": len(training_records("stage2"))},
        "total_optimization_steps_counting_pilots": sum(len(training_records(d)) for d in pilots + ["stage1", "stage2"]),
        "training_step_wall_seconds": training_seconds(pilots + ["stage1", "stage2"]),
        "target_steps_per_final_stage": target_steps,
    }, indent=2) + "\n")
    final1 = OUT / "stage1/checkpoint.pt"
    final2 = OUT / "stage2/checkpoint.pt"
    if not final1.exists():
        final1 = OUT / "pilot/stage1/checkpoint.pt"
    if not final2.exists():
        final2 = OUT / "pilot/stage2/checkpoint.pt"
    if final1.exists() and final2.exists():
        eval_cmd = [PYTHON, "scripts/evaluate_m17.py", "--stage1", str(final1),
                    "--stage2", str(final2), "--output-dir", str(OUT / "eval"),
                    "--dev-cap", "300", "--test-cap", "1260"]
        if run(eval_cmd, env) != 0:
            raise RuntimeError("final held-out evaluation failed")
        if run([PYTHON, "scripts/write_summary.py", "--run-dir", str(OUT)], env) != 0:
            raise RuntimeError("final research report generation failed")
    log("RUN_FINISHED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            log("RUN_FAILED " + repr(exc))
        finally:
            raise
