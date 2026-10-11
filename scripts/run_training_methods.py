#!/usr/bin/env python3
"""Train and evaluate one supported local method, or every supported method.

This deliberately delegates optimization and prediction to each method's own
maintained scripts. It does not replace native training loops with a generic one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "M10": {"path": "retrain/10_nimble_4b", "train": ["bash", "scripts/train.sh"], "eval": ["bash", "scripts/eval.sh"],
            "artifacts": ["results/dev.json", "results/dev_metric_predictions.jsonl", "results/dev_standard_metrics.json"],
            "metrics": "results/dev_standard_metrics.json",
            "checkpoint": "weights/nimble-4b-clean", "dataset": "Nimble family-disjoint inner dev"},
    "M11": {"path": "retrain/11_clm_style_4b", "train": ["bash", "scripts/train.sh"], "eval": ["bash", "scripts/eval.sh"],
            "artifacts": ["results/dev_predictions.jsonl", "results/dev_metric_predictions.jsonl", "results/dev_metrics.json"],
            "metrics": "results/dev_metrics.json",
            "checkpoint": "weights/clm-heads-clean", "dataset": "Nimble family-disjoint inner dev pairs"},
    "M12": {"path": "retrain/12_pointer_head_only_4b", "train": ["bash", "scripts/train.sh"], "eval": ["bash", "scripts/eval.sh"],
            "artifacts": ["results/dev_predictions.jsonl", "results/dev_metric_predictions.jsonl", "results/dev_metrics.json"],
            "metrics": "results/dev_metrics.json",
            "checkpoint": "weights/head_only-clean", "dataset": "Kev/Nimble family-disjoint inner dev"},
    "M13": {"path": "retrain/13_kev_fullft_4b", "train": ["bash", "scripts/train.sh"], "eval": ["bash", "scripts/eval.sh"],
            "artifacts": ["results/dev/predictions.jsonl", "results/dev/rows.json", "results/dev/report.json"],
            "metrics": "results/dev/report.json",
            "checkpoint": "weights/full_ft-clean", "dataset": "Kev/Nimble family-disjoint inner dev"},
    "M14": {"path": "retrain/14_vlm_systemone_4b", "train": ["bash", "scripts/train_v1.sh"],
            "eval": ["bash", "scripts/eval_inner.sh"], "artifacts": [], "metrics": "",
            "checkpoint": "", "dataset": "licensed image-decision train/model-validation splits"},
    "M15": {"path": "retrain/15_clef_style_4b", "train": ["bash", "scripts/train_supervised.sh"], "eval": ["bash", "scripts/eval.sh"],
            "artifacts": ["results/dev/predictions.jsonl", "results/dev/metrics.json"],
            "metrics": "results/dev/metrics.json",
            "checkpoint": "checkpoints/clef4b-head_only-clean", "dataset": "Nimble family-disjoint inner dev"},
}
M16 = {"path": "retrain/16_adas1_multiexit_4b", "dataset": "Nimble controlled P0 train/dev; 324 outer rows untouched"}
DEFERRED = {"M15-RLCD": "RLCD-like objective is explicitly deferred in scripts/train_rlcd.sh.",
            "M16-gate/serve/answerability": "Teacher and multi-exit head training are runnable; calibrated gates, open-set answerability, and true staged execution are still deferred."}


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def require_cuda(env: dict[str, str]) -> None:
    probe = subprocess.run(["python3", "-c", "import torch; print(torch.cuda.is_available())"],
                           cwd=ROOT, env=env, text=True, capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError("Cannot inspect the selected Python/PyTorch environment: " + probe.stderr.strip())
    if probe.stdout.strip().splitlines()[-1:] != ["True"]:
        raise RuntimeError("CUDA is unavailable to the selected PyTorch environment; no training/evaluation was started")


def metric_availability(method: str) -> dict:
    native_probability = method in {"M10", "M11", "M12", "M13", "M14", "M15"}
    task = "Native evaluation emits per-decision probability distributions on this method's inner-dev split."
    definitions = {
        "classification": (native_probability, ["accuracy", "macro/micro-F1", "per-class precision/recall/F1", "MRR", "Recall@3/5"], task),
        "routing": (False, ["Selected Success", "Regret", "Gain@Best", "Oracle Gap", "Routing Accuracy", "Recall@3/5", "MRR"],
                    "One model is evaluated per run; candidate-model routing outcomes are not recorded."),
        "decision_calibration": (native_probability, ["ECE", "Brier", "NLL", "reliability diagram"], task),
        "open_set": (False, ["AUROC", "FPR95", "AUOSCR/OpenAUC", "AUPR-Out", "Unknown F1"],
                     "The method dev split has no explicit known/unknown protocol."),
        "selective_decision": (native_probability, ["AURC", "Coverage@1% Risk", "Risk@80/90% Coverage"], task),
        "open_world": (False, ["AA", "AIA", "Average Forgetting", "BWT", "FWT", "Intransigence"],
                       "No sequential class-incremental train/eval matrix is produced."),
        "system_quality": (False, ["Task success", "Quality Retention"],
                           "No end-to-end task-success trace is collected by these method evals."),
        "cost": (False, ["Cost/request", "Cost/success", "CostSave@MatchedQuality", "API cost", "GPU amortization"],
                 "Cost and matched-quality traces are not recorded."),
        "latency": (method in {"M11", "M12", "M13", "M14", "M15"}, ["E2E p50", "E2E p95", "E2E p99", "router overhead ratio"],
                    "Timing comes from method-native inference; M13 has median and p95 only. Router overhead is not collected."),
        "compute": (False, ["GPU-sec/request", "Peak VRAM", "QPS/GPU"],
                    "GPU-seconds, peak VRAM, and QPS/GPU are not recorded."),
        "local_ai": (False, ["Local Serving Rate", "External Offload", "Escalation Rate"],
                     "Local-serving and offload decisions are not emitted by these model-only evals."),
        "multi_objective": (False, ["Pareto frontier", "ParetoDist", "normalized Utility"],
                            "Comparable quality, cost, latency, and GPU measurements are required."),
        "robustness": (False, ["OOD gap", "option permutation consistency", "long-context/modality shift"],
                       "No paired OOD, option-permutation, context-shift, or modality-shift evaluation is run here."),
    }
    return {"task_metrics": {"available": True, "scope": "native task metrics on method-specific inner dev", "note": task},
            "axes": {name: {"available": available, "metrics": names,
                            **({} if available else {"reason": reason})}
                     for name, (available, names, reason) in definitions.items()}}


def copy_artifacts(method: str, artifact_paths: list[str], out: Path) -> list[str]:
    copied = []
    method_root = ROOT / METHODS[method]["path"]
    for rel in artifact_paths:
        source = method_root / rel
        if source.exists():
            target = out / "native" / method / Path(rel).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                target = out / "native" / method / Path(rel).parent.name
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            copied.append(str(target.relative_to(out)))
    return copied


def latency_summary(method: str, method_root: Path, eval_dir: str | None = None):
    if method == "M10":
        return {"available": False, "reason": "M10 native evaluator does not record inference latency."}
    if method == "M13":
        path = method_root / str(eval_dir) / "report.json"
        native = json.loads(path.read_text(encoding="utf-8"))["latency_ms"]
        return {"available": True, "requests": None, "p50": native.get("median"),
                "p95": native.get("p95"), "p99": None,
                "unavailable": {"p99": "native evaluator does not report p99",
                                "router_overhead_ratio": "router timing is not recorded"},
                "source": "Kev native request timing"}
    if method == "M14":
        path = method_root / str(eval_dir) / "dev_predictions.jsonl"
    elif method == "M11":
        path = method_root / "results/dev_predictions.jsonl"
    elif method == "M12":
        path = method_root / "results/dev_predictions.jsonl"
    else:
        path = method_root / str(eval_dir) / "predictions.jsonl"
    values = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            latency = row.get("latency_ms")
            if isinstance(latency, (int, float)) and not isinstance(latency, bool) and math.isfinite(latency) and latency >= 0:
                values.append((str(row.get("request_id", row.get("id"))), float(latency)))
    if not values:
        return {"available": False, "reason": "No finite nonnegative native inference latency values were recorded."}
    by_request = {}
    for request_id, value in values:
        by_request.setdefault(request_id, value)
    ordered = sorted(by_request.values())
    def nearest_rank(q):
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"available": True, "requests": len(ordered), "p50": nearest_rank(.50),
            "p95": nearest_rank(.95), "p99": nearest_rank(.99),
            "unavailable": {"router_overhead_ratio": "router timing is not recorded"},
            "source": "method-native per-request inference timing"}


def run_method(method: str, out: Path, skip_train: bool = False, clef_stage: str = "head_only",
               m14_output: Path | None = None, m14_resume: bool = False) -> dict:
    spec = METHODS[method]
    method_root = ROOT / spec["path"]
    env = os.environ.copy()
    env["CONFIRM_TRAIN"] = "yes"
    env.setdefault("PYTHONUNBUFFERED", "1")
    venv_bin = ROOT / ".venv/bin"
    if venv_bin.is_dir():
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(ROOT / ".venv")
    if method == "M14":
        if skip_train and m14_output is None:
            raise ValueError("M14 --skip-train requires --m14-output pointing to an existing V1 run directory")
        target = m14_output if m14_output is not None else Path("results/training_runs") / out.name / "M14"
        target = target if target.is_absolute() else method_root / target
        target = target.resolve()
        if not target.is_relative_to(method_root.resolve()):
            raise ValueError("M14 output must remain inside retrain/14_vlm_systemone_4b")
        method_output = str(target.relative_to(method_root.resolve()))
        env["M14_OUTPUT"] = method_output
        if m14_resume:
            env["M14_RESUME"] = "yes"
        checkpoint_rel = f"{method_output}/best.pt"
        metric_rel = f"{method_output}/eval/shared_metrics.json"
        artifact_paths = [f"{method_output}/best.pt", f"{method_output}/last.pt",
                          f"{method_output}/progress.jsonl", f"{method_output}/run_config.json",
                          f"{method_output}/eval/dev_predictions.jsonl",
                          f"{method_output}/eval/metric_predictions.jsonl",
                          f"{method_output}/eval/shared_metrics.json"]
    elif method == "M15":
        env["CLEF_STAGE"] = clef_stage
        checkpoint_rel = f"checkpoints/clef4b-{clef_stage}-clean"
        metric_rel = "results/dev/metrics.json"
        artifact_paths = spec["artifacts"]
    else:
        checkpoint_rel = spec["checkpoint"]
        metric_rel = spec["metrics"]
    result = {"method": method, "directory": spec["path"], "dataset": spec["dataset"],
              "checkpoint": checkpoint_rel, "status": "running", "commands": [],
              "metric_coverage": metric_availability(method)}
    try:
        require_cuda(env)
        if not skip_train:
            command = spec["train"]
            result["commands"].append("train")
            run(command, method_root, env, out / "logs" / f"{method}-train.log")
        checkpoint = method_root / checkpoint_rel
        if not checkpoint.exists():
            raise FileNotFoundError(f"Expected checkpoint is missing: {checkpoint}")
        if spec["eval"]:
            result["commands"].append("evaluate")
            if method == "M14":
                result["eval_directory"] = f"{method_output}/eval"
                command = spec["eval"]
            elif method == "M15":
                eval_dir = f"results/dev_{out.name}"
                result["eval_directory"] = eval_dir
                command = [*spec["eval"], str(checkpoint_rel), eval_dir]
                artifact_paths = [f"{eval_dir}/predictions.jsonl", f"{eval_dir}/metrics.json"]
                metrics_rel = f"{eval_dir}/metrics.json"
            elif method == "M13":
                eval_dir = f"results/dev_{out.name}"
                env["KEV_EVAL_OUT"] = eval_dir
                result["eval_directory"] = eval_dir
                artifact_paths = [f"{eval_dir}/predictions.jsonl", f"{eval_dir}/rows.json",
                                  f"{eval_dir}/report.json", f"{eval_dir}/metric_predictions.jsonl",
                                  f"{eval_dir}/shared_metrics.json"]
                command = spec["eval"]
                metrics_rel = f"{eval_dir}/shared_metrics.json"
            else:
                artifact_paths = spec["artifacts"]
                command = spec["eval"]
                metrics_rel = metric_rel
            run(command, method_root, env, out / "logs" / f"{method}-eval.log")
            if method == "M14":
                run(["python3", "../../evaluation/standardize_training_eval.py", "M14",
                     "--native", f"{method_output}/eval/dev_predictions.jsonl",
                     "--output", f"{method_output}/eval/metric_predictions.jsonl"],
                    method_root, env, out / "logs" / f"{method}-normalize.log")
                run(["python3", "../../evaluation/metrics/compute_metrics.py",
                     f"{method_output}/eval/metric_predictions.jsonl", "--output",
                     f"{method_output}/eval/shared_metrics.json"],
                    method_root, env, out / "logs" / f"{method}-metrics.log")
            elif method == "M13":
                run(["python3", "../../evaluation/standardize_training_eval.py", "M13",
                     "--native", f"{eval_dir}/rows.json", "--output", f"{eval_dir}/metric_predictions.jsonl"],
                    method_root, env, out / "logs" / f"{method}-normalize.log")
                run(["python3", "../../evaluation/metrics/compute_metrics.py",
                     f"{eval_dir}/metric_predictions.jsonl", "--output", f"{eval_dir}/shared_metrics.json"],
                    method_root, env, out / "logs" / f"{method}-metrics.log")
        else:
            result["commands"].append("trainer_inner_dev_validation")
            if method != "M15":
                artifact_paths = spec["artifacts"]
        result["artifacts"] = copy_artifacts(method, artifact_paths, out)
        metrics_path = method_root / metrics_rel
        if metrics_path.is_file():
            result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            result["metric_artifact"] = str(metrics_path.relative_to(ROOT))
        else:
            raise FileNotFoundError(f"Expected evaluation metrics are missing: {metrics_path}")
        fallback_eval_dir = eval_dir if method in {"M13", "M15"} else None
        result["latency"] = latency_summary(method, method_root, result.get("eval_directory", fallback_eval_dir))
        result["status"] = "completed"
    except Exception as exc:  # preserve other methods' runs in --method all mode
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", nargs="?", default="all", help="M10, M11, M12, M13, M14, M15, M16, or all")
    parser.add_argument("--list", action="store_true", help="list runnable and deferred method IDs")
    parser.add_argument("--skip-train", action="store_true", help="evaluate an existing expected checkpoint")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/training_runs")
    parser.add_argument("--m14-output", type=Path,
                        help="existing M14 run directory relative to retrain/14_vlm_systemone_4b; used with --skip-train")
    parser.add_argument("--m14-resume", action="store_true",
                        help="resume M14 from --m14-output/last.pt using exactly the saved settings")
    parser.add_argument("--clef-stage", choices=("head_only", "lora"), default="head_only",
                        help="M15 supervised stage (also applies when method is all)")
    parser.add_argument("--stage", choices=("probe", "teacher", "multiexit", "full"),
                        help="M16 stage: P0 probe, full-depth teacher, multi-exit distillation, or teacher+distillation")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="trained M16 teacher head checkpoint required by --stage multiexit")
    parser.add_argument("--exit-layers", default="12,16,20",
                        help="candidate intermediate exits for M16 (default: 12,16,20)")
    args = parser.parse_args()
    if args.list:
        for name, spec in METHODS.items():
            print(f"runnable\t{name}\t{spec['path']}")
        for name, reason in DEFERRED.items():
            print(f"deferred\t{name}\t{reason}")
        for stage in ("probe", "teacher", "multiexit", "full"):
            print(f"runnable-stage\tM16\t{M16['path']}\t{stage}")
        return 0
    requested = args.method.upper()
    if requested not in {*METHODS, "M16", "ALL"}:
        parser.error(f"unknown method {args.method!r}; choose one of {', '.join([*METHODS, 'M16'])} or all")
    if requested == "M16" and args.stage is None:
        parser.error("M16 requires --stage {probe,teacher,multiexit,full}")
    if requested == "M16" and args.stage == "multiexit" and args.teacher_checkpoint is None:
        parser.error("M16 --stage multiexit requires --teacher-checkpoint")
    if requested != "M16" and (args.stage is not None or args.teacher_checkpoint is not None):
        parser.error("--stage and --teacher-checkpoint apply only to M16")
    if args.m14_output is not None and requested not in {"M14", "ALL"}:
        parser.error("--m14-output applies only to M14")
    if args.m14_resume and (requested not in {"M14", "ALL"} or args.skip_train or args.m14_output is None):
        parser.error("--m14-resume requires M14 (or all), --m14-output, and no --skip-train")
    if args.skip_train and requested in {"M14", "ALL"} and args.m14_output is None:
        parser.error("M14 --skip-train requires --m14-output")
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = args.output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    selected = list(METHODS) if requested == "ALL" else [requested]
    if requested == "M16":
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        venv_bin = ROOT / ".venv/bin"
        if venv_bin.is_dir():
            env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
            env["VIRTUAL_ENV"] = str(ROOT / ".venv")
        try:
            require_cuda(env)
            method_root = ROOT / M16["path"]
            stage = args.stage or "probe"
            if stage == "probe":
                output = method_root / "results/probes" / run_id
                output.parent.mkdir(parents=True, exist_ok=True)
                run(["python3", "scripts/prepare_data.py"], method_root, env, out / "logs/M16-prepare.log")
                run(["python3", "scripts/00_probe_all_layers.py", "--output", str(output)], method_root, env,
                    out / "logs/M16-probe.log")
                report_path = output / "probe_report.json"
            else:
                output = method_root / "results/training_runs" / run_id / stage
                output.parent.mkdir(parents=True, exist_ok=True)
                command = ["python3", "scripts/train_m16.py", "--stage", stage,
                           "--output", str(output), "--exit-layers", args.exit_layers]
                if args.teacher_checkpoint is not None:
                    command.extend(["--teacher-checkpoint", str(args.teacher_checkpoint)])
                run(command, method_root, env, out / f"logs/M16-{stage}.log")
                report_path = output / "run_report.json"
            if not report_path.is_file():
                raise FileNotFoundError(f"M16 {stage} did not produce {report_path}")
            report = {"run_id": run_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "protocol": f"M16 {stage}; controlled train/dev only; outer test never loaded",
                      "methods": [{"method": "M16", "stage": stage, "directory": M16["path"],
                                  "status": "completed", "run_report": str(report_path.relative_to(ROOT)),
                                  "metrics": json.loads(report_path.read_text(encoding="utf-8"))}],
                      "deferred_methods": DEFERRED, "status": "completed"}
        except Exception as exc:
            report = {"run_id": run_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "protocol": f"M16 {args.stage}; controlled train/dev only; outer test never loaded",
                      "methods": [{"method": "M16", "stage": args.stage, "directory": M16["path"],
                                  "status": "failed", "error": f"{type(exc).__name__}: {exc}"}],
                      "deferred_methods": DEFERRED, "status": "completed_with_failures"}
        (out / "standard_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(out / "standard_report.json")
        return 0 if report["status"] == "completed" else 1
    report = {"run_id": run_id, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "protocol": "method-native inner-dev evaluation; see metric_coverage per method",
              "methods": [run_method(method, out, args.skip_train, args.clef_stage,
                                     args.m14_output if method == "M14" else None,
                                     args.m14_resume if method == "M14" else False) for method in selected],
              "deferred_methods": DEFERRED}
    report["status"] = "completed" if all(row["status"] == "completed" for row in report["methods"]) else "completed_with_failures"
    (out / "standard_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out / "standard_report.json")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
