#!/usr/bin/env python3
"""Build the M17 report with per-item bootstrap intervals."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    conf, pred = probs.max(1), probs.argmax(1)
    error = (pred != labels).astype(float)
    brier = ((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1)
    nll = -np.log(np.maximum(1e-12, probs[np.arange(len(labels)), labels]))
    order = np.argsort(-conf, kind="stable")
    aurc = float((np.cumsum(error[order]) / np.arange(1, len(labels) + 1)).mean())
    ece = 0.0
    for i in range(10):
        mask = (conf >= i / 10) & (conf < (i + 1) / 10 if i < 9 else conf <= 1)
        if mask.any():
            ece += float(mask.mean() * abs(conf[mask].mean() - (1 - error[mask]).mean()))
    pred_risk = 1 - conf[order]
    pred_cum = np.cumsum(pred_risk) / np.arange(1, len(labels) + 1)
    eligible = np.flatnonzero(pred_cum <= 0.05)
    k = int(eligible[-1] + 1) if len(eligible) else 0
    return {"accuracy": float((pred == labels).mean()), "brier": float(brier.mean()),
            "nll": float(nll.mean()), "ece": ece, "aurc": aurc,
            "coverage_at_predicted_risk_5pct": k / len(labels),
            "realized_risk_at_that_coverage": float(error[order[:k]].mean()) if k else 0.0}


def bootstrap(labels, probs, reps=2000, seed=1147):
    rng = np.random.default_rng(seed)
    labels, probs = np.asarray(labels), np.asarray(probs)
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    values = {key: [] for key in metrics(labels, probs)}
    for index in draws:
        row = metrics(labels[index], probs[index])
        for key, val in row.items():
            values[key].append(val)
    return {key: {"low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5))}
            for key, vals in values.items()}


def paired_bootstrap(labels, candidate, base, reps=2000, seed=781):
    """Bootstrap candidate-minus-base metric deltas on shared test-item draws."""
    labels = np.asarray(labels, dtype=np.int64)
    candidate, base = np.asarray(candidate, dtype=float), np.asarray(base, dtype=float)
    if len(labels) != len(candidate) or candidate.shape != base.shape:
        raise ValueError("paired predictions must have the same test rows and class dimensions")
    point_a, point_b = metrics(labels, candidate), metrics(labels, base)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(reps, len(labels)))
    deltas = {key: [] for key in point_a}
    for index in draws:
        a = metrics(labels[index], candidate[index])
        b = metrics(labels[index], base[index])
        for key in deltas:
            deltas[key].append(a[key] - b[key])
    return {
        key: {
            "delta_candidate_minus_base": point_a[key] - point_b[key],
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
        }
        for key, values in deltas.items()
    }


def f4(value):
    return "—" if value is None else f"{value:.4f}"


def training_curves(root):
    """Export the main-stage optimization trace as portable CSV and SVG."""
    series = {}
    csv_rows = ["stage,step,training_seconds,loss,mean_correct_label_probability,mean_log_ratio,"
                "grad_norm,step_seconds,rationale_tokens,mean_rationale_tokens"]
    for stage in ("stage1", "stage2"):
        path = root / stage / "train.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
        elapsed = 0.0
        points = []
        for row in rows:
            elapsed += float(row["step_seconds"])
            point = (int(row["step"]), float(row["loss"]),
                     float(row["mean_correct_label_probability"]),
                     float(row.get("mean_log_ratio", 0.0)))
            points.append(point)
            csv_rows.append(",".join(str(x) for x in (
                stage, row["step"], f"{elapsed:.6f}", row["loss"],
                row["mean_correct_label_probability"], row.get("mean_log_ratio", 0.0),
                row["grad_norm"], row["step_seconds"], row["rationale_tokens"],
                row["mean_rationale_tokens"])))
        series[stage] = points

    (root / "training_curves.csv").write_text("\n".join(csv_rows) + "\n")
    width, height = 960, 660
    left, right = 78, 930
    panels = [(72, 238, "Objective loss"), (290, 456, "Mean probability of observed label"),
              (508, 620, "Stage 2 sampled policy/reference log ratio")]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.s1{fill:none;stroke:#2563eb;stroke-width:2}.s2{fill:none;stroke:#dc2626;stroke-width:2}</style>',
           '<text x="480" y="30" text-anchor="middle" font-size="18" font-weight="bold">M17 main-stage training curves</text>']
    for panel_idx, (top, bottom, title) in enumerate(panels):
        svg += [f'<text x="{left}" y="{top - 14}" font-size="13">{title}</text>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>']
        keys = {0: 1, 1: 2, 2: 3}
        visible_series = series.values() if panel_idx != 2 else (series["stage2"],)
        all_values = [p[keys[panel_idx]] for values in visible_series for p in values]
        if panel_idx == 2:
            all_values = [p[3] for p in series["stage2"]]
        if not all_values:
            svg.append(f'<text x="{(left + right) // 2}" y="{(top + bottom) // 2}" text-anchor="middle" font-size="12">No main-stage steps recorded</text>')
            continue
        lo, hi = min(all_values), max(all_values)
        if hi - lo < 1e-9:
            pad = max(abs(hi) * 0.05, 0.05)
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.08
            lo, hi = lo - pad, hi + pad
        max_step = max((p[0] for values in series.values() for p in values), default=1)
        for tick in range(5):
            y = bottom - (bottom - top) * tick / 4
            value = lo + (hi - lo) * tick / 4
            svg += [f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>',
                    f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{value:.3g}</text>']
        for stage, points in series.items():
            if panel_idx == 2 and stage != "stage2":
                continue
            vals = [(p[0], p[keys[panel_idx]]) for p in points]
            if not vals:
                continue
            coords = [f'{left + (right - left) * (step - 1) / max(1, max_step - 1):.1f},'
                      f'{bottom - (value - lo) * (bottom - top) / (hi - lo):.1f}'
                      for step, value in vals]
            css_class = "s1" if stage == "stage1" else "s2"
            point_text = " ".join(coords)
            svg.append(f'<polyline class="{css_class}" points="{point_text}"/>')
        svg += [f'<text x="{left}" y="{bottom + 16}" font-size="10">1</text>',
                f'<text x="{right}" y="{bottom + 16}" text-anchor="end" font-size="10">{max_step}</text>',
                f'<text x="{(left + right) // 2}" y="{bottom + 31}" text-anchor="middle" font-size="11">Optimization step</text>']
    svg += ['<line x1="720" y1="44" x2="746" y2="44" stroke="#2563eb" stroke-width="3"/><text x="752" y="48" font-size="11">Stage 1</text>',
            '<line x1="820" y1="44" x2="846" y2="44" stroke="#dc2626" stroke-width="3"/><text x="852" y="48" font-size="11">Stage 2</text>',
            '</svg>']
    (root / "training_curves.svg").write_text("\n".join(svg) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    root = Path(ap.parse_args().run_dir)
    training_curves(root)
    arms = {name: json.loads((root / "eval" / f"{name}.json").read_text())
            for name in ("base", "stage1", "stage2")}
    if len({tuple(a["test_uids"]) for a in arms.values()}) != 1:
        raise SystemExit("evaluation arms do not share exact test UIDs")
    labels = arms["base"]["test_labels"]
    if any(a["test_labels"] != labels for a in arms.values()):
        raise SystemExit("evaluation arms do not share exact labels")
    results = {}
    scaled_predictions = {}
    for name, arm in arms.items():
        raw = arm["predictions"]["test_single"]
        temperature = arm["readouts"]["single"]["temperature"]
        z = np.log(np.maximum(1e-12, np.asarray(raw, dtype=float))) / temperature
        scaled = np.exp(z - z.max(1, keepdims=True))
        scaled /= scaled.sum(1, keepdims=True)
        scaled_predictions[name] = scaled
        result = metrics(labels, scaled)
        result["confidence_intervals_95pct"] = bootstrap(labels, scaled)
        result["temperature"] = temperature
        result["temperature_scaled_metrics_source"] = arm["readouts"]["single"]["temperature_scaled"]
        result["raw_metrics_source"] = arm["readouts"]["single"]["raw"]
        results[name] = result
    (root / "eval/bootstrap_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    paired = {name: paired_bootstrap(labels, scaled_predictions[name], scaled_predictions["base"],
                                     seed=781 + index)
              for index, name in enumerate(("stage1", "stage2"), start=1)}
    (root / "eval/paired_comparisons.json").write_text(json.dumps(paired, indent=2) + "\n")
    steps = json.loads((root / "training_steps.json").read_text())
    config = json.loads((root / "effective_config.json").read_text())
    complete = (steps["final"]["stage1"] == 400 and steps["final"]["stage2"] == 400
                and (root / "stage1/checkpoint.pt").is_file()
                and (root / "stage2/checkpoint.pt").is_file()
                and (root / "preflight.json").is_file())
    keys = ["accuracy", "brier", "nll", "ece", "aurc",
            "coverage_at_predicted_risk_5pct", "realized_risk_at_that_coverage"]
    table = ["| Arm | Acc. | Brier | NLL | ECE | AURC | Cov@5% | Realized risk |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        table.append("| " + name + " | " + " | ".join(f4(result[k]) for k in keys) + " |")
    paired_keys = ["accuracy", "brier", "nll", "ece", "aurc",
                   "coverage_at_predicted_risk_5pct", "realized_risk_at_that_coverage"]
    paired_table = ["| Comparison | " + " | ".join(paired_keys) + " |",
                    "|---|" + "---:|" * len(paired_keys)]
    for name, comparison in paired.items():
        cells = []
        for key in paired_keys:
            delta = comparison[key]
            cells.append(f"{delta['delta_candidate_minus_base']:.4f} "
                         f"[{delta['ci95_low']:.4f}, {delta['ci95_high']:.4f}]")
        paired_table.append(f"| {name} − base | " + " | ".join(cells) + " |")
    stage_lines = []
    for stage in ("stage1", "stage2"):
        path = root / stage / "train.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []
        durations = [r["step_seconds"] for r in rows]
        tokens = sum(r["rationale_tokens"] for r in rows)
        p90 = float(np.percentile(durations, 90)) if durations else None
        peak = max((r["peak_memory_allocated_bytes"] for r in rows), default=0)
        stage_lines.append(f"- {stage}: {len(rows)} main steps, {tokens} rationale tokens; "
                           f"step mean/p50/p90 {f4(statistics.mean(durations) if durations else None)}/"
                           f"{f4(statistics.median(durations) if durations else None)}/{f4(p90)} sec; "
                           f"peak allocated VRAM {peak} bytes.")
    report = [
        "# M17 OpenJev-RLCD on Qwen3.5-4B — experiment report", "",
        "**Status: TRAINING COMPLETED — adapted single-seed pilot.**" if complete else "**Status: PARTIAL.**",
        "",
        "## Provenance and configuration", "",
        "- Upstream commit: `22db40ad3333f5cf0bc5139097da70e0f8c05018`.",
        f"- Model: `{config['model']}` revision `{config['model_revision']}`.",
        "- Original GSM8K-Verify file; split seed 0; 4,000 train, 500 dev, 1,260 original GSM8K test rows. See dataset manifest for SHA and UID hashes.",
        f"- Main steps: Stage 1 {steps['final']['stage1']}, Stage 2 {steps['final']['stage2']}; {steps['total_optimization_steps_counting_pilots']} steps including smoke, LR tuning and timed pilots.",
        f"- Optimization-step time: {steps['training_step_wall_seconds'] / 3600:.3f} GPU-hours, including smoke and pilots; VKong billing/provisioning and final evaluation are separate.",
        f"- LR selected on first 64 dev UIDs by raw Brier (NLL tie-break): Stage 1 {config.get('stage1_lr')}, Stage 2 {config.get('stage2_lr')}. This subset shares the dev pool used for temperature calibration.",
        "- Objective: individual-rationale Brier-derived score, diversity weight 0; Stage 1 pathwise only; Stage 2 pathwise plus leave-one-out REINFORCE (`sf_coef=0.3`) and frozen Stage-1 KL reference (`beta=0.04`).", "",
        "## Held-out results", "",
        f"All arms use the same {len(labels)} test UIDs. Temperature is fit on the same 300-item dev pool for each arm. Intervals are 95% percentile bootstrap intervals over test items.", "",
        *table, "",
        "Paired candidate-minus-base differences after dev temperature scaling (95% percentile bootstrap CI; each draw resamples the same test UIDs for both arms):", "",
        *paired_table, "",
        "Realized risk is the observed error among the largest confidence-ranked subset whose predicted error is at most 5%; this threshold does not guarantee 5% realized error. Full confidence intervals and per-item predictions are in `eval/`.", "",
        "## Efficiency and artifacts", "", *stage_lines, "",
        "- Throughput projection: `throughput_projection.json`.",
        "- Main-stage training curves: `training_curves.svg` and source data `training_curves.csv`.",
        "- Effective config and dev-only LR screen: `effective_config.json`, `learning_rate_selection.json`.",
        "- Environment: `environment.json`, `environment.txt`, `gpu.txt`; dataset and source hashes under `manifests/`.",
        "- Resumable checkpoints (when present): `stage1/checkpoint.pt`, `stage2/checkpoint.pt`; include LoRA, optimizer, RNG/sampler RNG, step, elapsed time, config and reference digest.",
        "- Logs: `run.log`, stage `train.jsonl`; eval files: `eval/base.json`, `eval/stage1.json`, `eval/stage2.json`, `eval/comparison.json`, `eval/bootstrap_metrics.json`, `eval/paired_comparisons.json`.", "",
        "## Adaptations and limits", "",
        "- This is an independent OpenJev-RLCD adaptation, not a reproduction of TypeSafe Jev internals.",
        "- Qwen3.5-4B uses the official multimodal checkpoint in BF16 with unused vision weights omitted after text-logit equivalence is checked; rank-16 LoRA replaces the original full FP32 optimizer model.",
        "- The generation policy and Stage-2 frozen reference share one base model via separate adapters. Stage 1 updates only LoRA weights; Stage 2 retains pathwise gradients and the frozen Stage-1 KL anchor.",
        "- One seed pair is a pilot, not the paper's three-seed statistical study or a controlled cross-method comparison.",
        "- Treat reported test results as invalid for a completion claim if any GPU preflight, training, checkpoint-load or evaluation gate failed.", "",
    ]
    out = root / "summary.md"
    out.write_text("\n".join(report))
    print(out)


if __name__ == "__main__":
    main()
