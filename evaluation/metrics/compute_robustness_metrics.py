#!/usr/bin/env python3
"""Compute OOD accuracy gap and paired option-permutation consistency.

Rows use the common prediction shape and may also carry `domain_group` with
values `in_domain`/`ood`, plus `pair_id`, `variant_role` (`baseline` or
`variant`), and optional `choice_semantics` mapping presented option IDs to
canonical class IDs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from compute_metrics import validate_row


def load(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                rows.append(validate_row(json.loads(line), path, line_no))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load(args.predictions)
    result = {}
    domain_groups = defaultdict(list)
    for row in rows:
        group = row.get("domain_group", row.get("metadata", {}).get("domain_group"))
        if group is not None:
            domain_groups[str(group)].append(row)
    if "in_domain" in domain_groups and "ood" in domain_groups:
        accuracies = {key: sum(max(r["probabilities"], key=r["probabilities"].get) == r["label"]
                               for r in values) / len(values)
                      for key, values in domain_groups.items() if key in {"in_domain", "ood"}}
        result["ood_accuracy"] = accuracies
        result["ood_gap"] = accuracies["in_domain"] - accuracies["ood"]
        result["ood_gap_definition"] = "in-domain accuracy minus OOD accuracy; positive means OOD degradation"
    else:
        result["ood_gap"] = None
        result["ood_gap_unavailable_reason"] = "rows must include domain_group=in_domain and domain_group=ood"

    pairs = defaultdict(list)
    for row in rows:
        pair_id = row.get("pair_id", row.get("metadata", {}).get("pair_id"))
        role = row.get("variant_role", row.get("metadata", {}).get("variant_role"))
        if pair_id is not None and role is not None:
            pairs[str(pair_id)].append((role, row))
    comparisons = consistent = 0
    for pair_id, members in pairs.items():
        baselines = [row for role, row in members if role == "baseline"]
        variants = [row for role, row in members if role == "variant"]
        if len(baselines) != 1 or not variants:
            raise ValueError(f"pair {pair_id} must have exactly one baseline and at least one variant")
        baseline = baselines[0]
        base_choice = max(baseline["probabilities"], key=baseline["probabilities"].get)
        base_map = baseline.get("choice_semantics", {})
        if base_choice not in base_map and base_map:
            raise ValueError(f"pair {pair_id}: baseline prediction missing from choice_semantics")
        base_semantic = base_map.get(base_choice, base_choice)
        base_universe = set(base_map.values()) if base_map else set(baseline["probabilities"])
        for variant in variants:
            choice = max(variant["probabilities"], key=variant["probabilities"].get)
            choice_map = variant.get("choice_semantics", {})
            if choice_map and choice not in choice_map:
                raise ValueError(f"pair {pair_id}: variant prediction missing from choice_semantics")
            variant_universe = set(choice_map.values()) if choice_map else set(variant["probabilities"])
            if variant_universe != base_universe:
                raise ValueError(f"pair {pair_id}: baseline and variant must contain the same semantic options")
            consistent += base_semantic == choice_map.get(choice, choice)
            comparisons += 1
    if comparisons:
        result["option_permutation_consistency"] = consistent / comparisons
        result["option_permutation_flip_rate"] = 1 - consistent / comparisons
        result["option_permutation_pairs"] = len(pairs)
        result["option_permutation_comparisons"] = comparisons
    else:
        result["option_permutation_consistency"] = None
        result["option_permutation_unavailable_reason"] = (
            "paired rows need pair_id and variant_role=baseline/variant; map shuffled option IDs "
            "to canonical labels using choice_semantics"
        )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
