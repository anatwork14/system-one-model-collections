#!/usr/bin/env python3
"""Score a router from per-request outcomes for all candidate models.

Each JSONL row needs `id`, `selected_model`, and `candidate_outcomes`, a map
from model name to {"success": float, optional "utility": float}. Optional
`route_scores` maps model names to the router's pre-selection scores.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def finite_unit(value, name, row_id):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{row_id}: {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{row_id}: {name} must be finite and in [0,1]")
    return value


def load(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"{path}:{number}: id is required")
            outcomes = row.get("candidate_outcomes")
            selected = row.get("selected_model")
            if not isinstance(outcomes, dict) or not outcomes or selected not in outcomes:
                raise ValueError(f"{path}:{number}: selected_model must exist in candidate_outcomes")
            for model, outcome in outcomes.items():
                if not isinstance(outcome, dict) or "success" not in outcome:
                    raise ValueError(f"{path}:{number}: {model} needs a success outcome")
                outcome["success"] = finite_unit(outcome["success"], f"{model}.success", row["id"])
                if "utility" in outcome:
                    value = outcome["utility"]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"{path}:{number}: {model}.utility must be finite numeric")
                    outcome["utility"] = float(value)
            scores = row.get("route_scores")
            if scores is not None:
                if not isinstance(scores, dict) or set(scores) != set(outcomes):
                    raise ValueError(f"{path}:{number}: route_scores must score every candidate model exactly once")
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in scores.values()):
                    raise ValueError(f"{path}:{number}: route scores must be finite numeric values")
            row["candidate_outcomes"] = outcomes
            rows.append(row)
    if not rows:
        raise ValueError(f"No routing rows in {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("routing IDs must be unique")
    model_sets = {tuple(sorted(row["candidate_outcomes"])) for row in rows}
    if len(model_sets) != 1:
        raise ValueError("every request must have outcomes for the same candidate models")
    return rows


def summarize(rows, recall_ks):
    models = sorted(rows[0]["candidate_outcomes"])
    selected_success = selected_utility = oracle_utility = oracle_success = 0.0
    selected_correct = route_count = 0
    recall_hits = {k: 0 for k in recall_ks}
    reciprocal_rank = 0.0
    has_route_scores = all("route_scores" in row for row in rows)
    fixed_success = {model: 0.0 for model in models}
    fixed_utility = {model: 0.0 for model in models}
    for row in rows:
        outcomes = row["candidate_outcomes"]
        for model in models:
            fixed_success[model] += outcomes[model]["success"]
            fixed_utility[model] += outcomes[model].get("utility", outcomes[model]["success"])
        chosen = outcomes[row["selected_model"]]
        utility = {model: outcomes[model].get("utility", outcomes[model]["success"]) for model in models}
        top_utility = max(utility.values())
        best_models = {model for model, value in utility.items() if math.isclose(value, top_utility, abs_tol=1e-12)}
        selected_success += chosen["success"]
        selected_utility += utility[row["selected_model"]]
        oracle_utility += top_utility
        oracle_success += max(outcome["success"] for outcome in outcomes.values())
        selected_correct += row["selected_model"] in best_models
        route_count += 1
        if has_route_scores:
            ranked = sorted(row["route_scores"], key=row["route_scores"].get, reverse=True)
            rank = next(index for index, model in enumerate(ranked, 1) if model in best_models)
            reciprocal_rank += 1 / rank
            for k in recall_ks:
                recall_hits[k] += bool(best_models.intersection(ranked[:k]))
    n = len(rows)
    fixed_mean = {model: fixed_success[model] / n for model in models}
    best_fixed = max(fixed_mean.values())
    result = {
        "requests": n,
        "selected_success": selected_success / n,
        "mean_selected_utility": selected_utility / n,
        "mean_oracle_utility": oracle_utility / n,
        "regret": (oracle_utility - selected_utility) / n,
        "oracle_gap": (oracle_success - selected_success) / n,
        "best_fixed_model": max(fixed_mean, key=fixed_mean.get),
        "best_fixed_success": best_fixed,
        "gain_at_best": selected_success / n - best_fixed,
        "gain_over_best_fixed_model": selected_success / n - best_fixed,
        "fixed_model_success": fixed_mean,
        "router_accuracy": selected_correct / route_count if route_count else None,
        "recall_at_k": {str(k): recall_hits[k] / n if has_route_scores else None for k in recall_ks},
        "mrr": reciprocal_rank / n if has_route_scores else None,
        "route_ranking_available": has_route_scores,
        "definitions": {
            "regret": "per-request best candidate utility minus selected candidate utility; utility defaults to success",
            "oracle_gap": "per-request oracle task success minus selected task success",
            "gain_over_best_fixed_model": "selected mean success minus the best single candidate model mean success",
            "router_accuracy": "fraction where selected model is an oracle-utility maximizer (ties count correct)",
            "recall_at_k_mrr": "rank oracle-utility maximizers by route_scores; ties in actual utility count as relevant",
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(k < 1 for k in args.recall_k):
        parser.error("recall k values must be positive")
    rows = load(args.predictions)
    result = summarize(rows, sorted(set(args.recall_k)))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
