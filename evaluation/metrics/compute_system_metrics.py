#!/usr/bin/env python3
"""Summarize per-request serving traces and method-level Pareto tradeoffs.

Required row fields: method, id, task_success (0..1), latency_ms. Optional
fields: cost_usd, gpu_seconds, peak_vram_mb, router_latency_ms,
served_locally, external_offload, escalated, run_id, gpu_wall_seconds, num_gpus.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def quantile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def number(row, key, optional=False):
    value = row.get(key)
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{row.get('id')}: {key} must be a finite number")
    if value < 0:
        raise ValueError(f"{row.get('id')}: {key} cannot be negative")
    return float(value)


def load(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id") or not row.get("method"):
                raise ValueError(f"{path}:{line_no}: method and id are required")
            row["task_success"] = number(row, "task_success")
            if row["task_success"] > 1:
                raise ValueError(f"{row['id']}: task_success must be in [0,1]")
            row["latency_ms"] = number(row, "latency_ms")
            for key in ("cost_usd", "gpu_seconds", "peak_vram_mb", "router_latency_ms", "gpu_wall_seconds", "num_gpus"):
                if key in row:
                    row[key] = number(row, key)
            if "num_gpus" in row and (row["num_gpus"] < 1 or not row["num_gpus"].is_integer()):
                raise ValueError(f"{row['id']}: num_gpus must be a positive integer")
            if "router_latency_ms" in row and row["router_latency_ms"] > row["latency_ms"]:
                raise ValueError(f"{row['id']}: router latency cannot exceed end-to-end latency")
            for key in ("served_locally", "external_offload", "escalated"):
                if key in row and not isinstance(row[key], bool):
                    raise ValueError(f"{row['id']}: {key} must be boolean")
            rows.append(row)
    if not rows:
        raise ValueError(f"No system trace rows in {path}")
    return rows


def summarize(rows):
    n = len(rows)
    successes = sum(row["task_success"] for row in rows)
    costs = [row["cost_usd"] for row in rows if "cost_usd" in row]
    gpu_seconds = [row["gpu_seconds"] for row in rows if "gpu_seconds" in row]
    vram = [row["peak_vram_mb"] for row in rows if "peak_vram_mb" in row]
    router_ratio = [row["router_latency_ms"] / row["latency_ms"]
                    for row in rows if "router_latency_ms" in row and row["latency_ms"] > 0]
    coverage = {
        "cost_usd": len(costs), "gpu_seconds": len(gpu_seconds), "peak_vram_mb": len(vram),
        "router_latency_ms": sum("router_latency_ms" in row for row in rows),
        "gpu_wall_seconds": sum("gpu_wall_seconds" in row for row in rows),
        "num_gpus": sum("num_gpus" in row for row in rows),
        "served_locally": sum("served_locally" in row for row in rows),
        "external_offload": sum("external_offload" in row for row in rows),
        "escalated": sum("escalated" in row for row in rows),
    }
    result = {
        "requests": n,
        "task_success": successes / n,
        "latency_ms": {f"p{int(q * 100)}": quantile([r["latency_ms"] for r in rows], q)
                       for q in (0.50, 0.95, 0.99)},
        "router_overhead_ratio_mean": sum(router_ratio) / len(router_ratio) if len(router_ratio) == n else None,
        "cost_usd_per_request": sum(costs) / n if len(costs) == n else None,
        "cost_usd_per_success": sum(costs) / successes if len(costs) == n and successes else None,
        "gpu_seconds_per_request": sum(gpu_seconds) / n if len(gpu_seconds) == n else None,
        "peak_vram_mb": max(vram) if len(vram) == n else None,
        "local_serving_rate": sum(row.get("served_locally") is True for row in rows) / n
                              if coverage["served_locally"] == n else None,
        "external_offload_rate": sum(row.get("external_offload") is True for row in rows) / n
                                 if coverage["external_offload"] == n else None,
        "escalation_rate": sum(row.get("escalated") is True for row in rows) / n
                           if coverage["escalated"] == n else None,
        "qps_per_gpu": None,
        "measurement_coverage": coverage,
    }
    run_walls = {}
    run_gpus = {}
    for row in rows:
        if "gpu_wall_seconds" not in row:
            continue
        run_id = row.get("run_id")
        if not run_id or "num_gpus" not in row:
            raise ValueError("gpu_wall_seconds requires run_id and num_gpus for per-GPU throughput")
        previous = run_walls.setdefault(run_id, row["gpu_wall_seconds"])
        if not math.isclose(previous, row["gpu_wall_seconds"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"inconsistent gpu_wall_seconds for run {run_id}")
        previous_gpus = run_gpus.setdefault(run_id, row["num_gpus"])
        if previous_gpus != row["num_gpus"]:
            raise ValueError(f"inconsistent num_gpus for run {run_id}")
    wall_seconds = sum(run_walls[run_id] * run_gpus[run_id] for run_id in run_walls)
    if (wall_seconds > 0 and coverage["gpu_wall_seconds"] == n and coverage["num_gpus"] == n
            and len(run_walls) == len({row.get("run_id") for row in rows})):
        result["qps_per_gpu"] = n / wall_seconds
    return result


def pareto(method_metrics, weights):
    candidates = {name: value for name, value in method_metrics.items()
                  if value["cost_usd_per_request"] is not None
                  and value["gpu_seconds_per_request"] is not None}
    if not candidates:
        return {"methods": {}, "unavailable_reason": "cost and GPU time are needed for Pareto analysis"}
    dimensions = {
        "quality_loss": {name: 1.0 - metrics["task_success"] for name, metrics in candidates.items()},
        "cost": {name: metrics["cost_usd_per_request"] for name, metrics in candidates.items()},
        "latency": {name: metrics["latency_ms"]["p50"] for name, metrics in candidates.items()},
        "gpu_seconds": {name: metrics["gpu_seconds_per_request"] for name, metrics in candidates.items()},
    }
    mins = {key: min(values.values()) for key, values in dimensions.items()}
    maxs = {key: max(values.values()) for key, values in dimensions.items()}
    normalized = {name: {key: (value - mins[key]) / (maxs[key] - mins[key]) if maxs[key] > mins[key] else 0.0
                         for key, values in dimensions.items() for value in [values[name]]}
                  for name in candidates}
    front = {}
    for name, vector in normalized.items():
        dominated = any(other != name and all(normalized[other][key] <= vector[key] for key in vector)
                        and any(normalized[other][key] < vector[key] for key in vector)
                        for other in normalized)
        front[name] = not dominated
    frontier_points = [normalized[name] for name, is_front in front.items() if is_front]
    utility = None
    if weights:
        if set(weights) != set(dimensions) or any(isinstance(weight, bool) or not isinstance(weight, (int, float))
                                                  or weight < 0 or not math.isfinite(weight)
                                                  for weight in weights.values()):
            raise ValueError("utility weights must provide nonnegative values for quality_loss,cost,latency,gpu_seconds")
        if sum(weights.values()) <= 0:
            raise ValueError("utility weights must sum to a positive number")
        weight_total = sum(weights.values())
        utility = {name: 1.0 - sum(weights[key] * normalized[name][key] for key in weights) / weight_total
                   for name in candidates}
    return {"objectives": "maximize task_success; minimize cost/request, p50 latency, GPU-sec/request",
            "pareto_front": [name for name, is_front in front.items() if is_front],
            "normalized_objectives": normalized,
            "pareto_distance": {name: min(math.sqrt(sum((vector[key] - point[key]) ** 2 for key in vector))
                                           for point in frontier_points)
                                for name, vector in normalized.items()},
            "distance_to_ideal": {name: math.sqrt(sum(x * x for x in vector.values()))
                                  for name, vector in normalized.items()},
            "normalized_utility": utility,
            "utility_direction": "1 minus weighted normalized loss; supply --utility-weights for policy weights",
            "normalization_bounds": {key: {"min": mins[key], "max": maxs[key]} for key in mins}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--reference-method")
    parser.add_argument("--quality-floor", type=float,
                        help="minimum task success for CostSave@MatchedQuality against reference")
    parser.add_argument("--utility-weights", help="JSON weights for quality_loss,cost,latency,gpu_seconds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.reference_method is None) != (args.quality_floor is None):
        parser.error("--reference-method and --quality-floor must be supplied together")
    if args.quality_floor is not None and not 0 <= args.quality_floor <= 1:
        parser.error("--quality-floor must be in [0,1]")
    rows = load(args.traces)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    id_sets = {name: {row["id"] for row in method_rows} for name, method_rows in grouped.items()}
    if len(id_sets) > 1 and len({frozenset(ids) for ids in id_sets.values()}) != 1:
        parser.error("method comparisons require identical request IDs for every method")
    for name, method_rows in grouped.items():
        ids = [row["id"] for row in method_rows]
        if len(ids) != len(set(ids)):
            parser.error(f"duplicate request ID for method {name}")
    metrics = {name: summarize(method_rows) for name, method_rows in sorted(grouped.items())}
    best_quality = max(summary["task_success"] for summary in metrics.values())
    for summary in metrics.values():
        summary["quality_retention_vs_best_method"] = (
            summary["task_success"] / best_quality if best_quality else None
        )
    if args.reference_method:
        if args.reference_method not in metrics:
            parser.error(f"reference method {args.reference_method!r} not found")
        reference = metrics[args.reference_method]
        if reference["cost_usd_per_request"] is None:
            parser.error("reference method has no cost_usd values")
        if reference["task_success"] < args.quality_floor:
            parser.error("reference method does not meet the requested quality floor")
        for name, summary in metrics.items():
            if summary["task_success"] >= args.quality_floor and summary["cost_usd_per_request"] is not None:
                summary["cost_save_at_matched_quality"] = (
                    reference["cost_usd_per_request"] - summary["cost_usd_per_request"]
                ) / reference["cost_usd_per_request"] if reference["cost_usd_per_request"] else None
            else:
                summary["cost_save_at_matched_quality"] = None
        metrics[args.reference_method]["matched_quality_floor"] = args.quality_floor
    weights = json.loads(args.utility_weights) if args.utility_weights else None
    result = {"methods": metrics, "multi_objective": pareto(metrics, weights)}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
