#!/usr/bin/env python3
"""Compute classification, calibration, selective, and open-set metrics.

Input is one JSON object per decision with `id`, `label`, and a normalized
`probabilities` map. Optional metadata fields are retained in slice summaries.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path, bins: int):
    if bins < 1:
        raise ValueError("--bins must be positive")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(validate_row(row, path, line_number))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction IDs must be unique within a metric input")
    return rows


def validate_row(row, path, line_number):
    where = f"{path}:{line_number}"
    if not isinstance(row, dict) or not row.get("id") or "label" not in row:
        raise ValueError(f"{where}: each row needs a nonempty id and label")
    probs = row.get("probabilities")
    if not isinstance(probs, dict) or len(probs) < 2:
        raise ValueError(f"{where}: probabilities must map at least two option IDs to values")
    values = {}
    for key, value in probs.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: probability entries must have string keys and numeric values")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{where}: probabilities must be finite and in [0,1]")
        values[key] = value
    if abs(sum(values.values()) - 1.0) > 1e-4:
        raise ValueError(f"{where}: probabilities must sum to 1 within 1e-4")
    label = "true" if row["label"] is True else "false" if row["label"] is False else str(row["label"])
    if label not in values:
        raise ValueError(f"{where}: label {label!r} missing from probabilities")
    predicted = max(values, key=values.get)
    if "predicted_option" in row and str(row["predicted_option"]) != predicted:
        raise ValueError(f"{where}: predicted_option does not match the probability argmax")
    row["label"] = label
    row["probabilities"] = values
    return row


def ece_and_reliability(rows, bins):
    buckets = [{"count": 0, "confidence_sum": 0.0, "correct_sum": 0.0} for _ in range(bins)]
    for row in rows:
        probs = row["probabilities"]
        pred = max(probs, key=probs.get)
        confidence = probs[pred]
        index = min(int(confidence * bins), bins - 1)
        bucket = buckets[index]
        bucket["count"] += 1
        bucket["confidence_sum"] += confidence
        bucket["correct_sum"] += float(pred == row["label"])
    total = len(rows)
    ece = 0.0
    reliability = []
    for index, bucket in enumerate(buckets):
        count = bucket["count"]
        low, high = index / bins, (index + 1) / bins
        if count:
            confidence = bucket["confidence_sum"] / count
            accuracy = bucket["correct_sum"] / count
            ece += count / total * abs(confidence - accuracy)
            reliability.append({"bin": index, "lower": low, "upper": high, "count": count,
                                "mean_confidence": confidence, "accuracy": accuracy})
        else:
            reliability.append({"bin": index, "lower": low, "upper": high, "count": 0,
                                "mean_confidence": None, "accuracy": None})
    return ece, reliability


def risk_coverage(rows, coverage):
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: max(row["probabilities"].values()), reverse=True)
    take = max(1, math.ceil(coverage * len(ordered)))
    errors = sum(max(row["probabilities"], key=row["probabilities"].get) != row["label"]
                 for row in ordered[:take])
    return {"coverage": take / len(ordered), "risk": errors / take, "accepted": take}


def coverage_at_risk(rows, maximum_risk):
    ordered = sorted(rows, key=lambda row: max(row["probabilities"].values()), reverse=True)
    errors = accepted = best_accepted = 0
    for row in ordered:
        errors += max(row["probabilities"], key=row["probabilities"].get) != row["label"]
        accepted += 1
        if errors / accepted <= maximum_risk:
            best_accepted = accepted
    return {"coverage": best_accepted / len(rows), "accepted": best_accepted,
            "maximum_risk": maximum_risk}


def binary_ranking(positive_scores, negative_scores):
    """Return AUROC, average precision, and FPR at >=95% recall for positives."""
    positives, negatives = len(positive_scores), len(negative_scores)
    if not positives or not negatives:
        return {"auroc": None, "aupr_out": None, "fpr95": None,
                "unavailable_reason": "both positive and negative examples are required"}
    scored = [(float(score), 1) for score in positive_scores] + [(float(score), 0) for score in negative_scores]
    scored.sort(key=lambda item: item[0], reverse=True)
    tp = fp = 0
    roc = [(0.0, 0.0)]
    ap = 0.0
    fpr95 = None
    index = 0
    while index < len(scored):
        threshold = scored[index][0]
        while index < len(scored) and scored[index][0] == threshold:
            tp += scored[index][1]
            fp += 1 - scored[index][1]
            index += 1
        tpr, fpr = tp / positives, fp / negatives
        roc.append((fpr, tpr))
        previous_tpr = roc[-2][1]
        ap += (tpr - previous_tpr) * (tp / (tp + fp))
        if fpr95 is None and tpr >= 0.95:
            fpr95 = fpr
    auroc = sum((x2 - x1) * (y1 + y2) / 2
                for (x1, y1), (x2, y2) in zip(roc, roc[1:]))
    return {"auroc": auroc, "aupr_out": ap, "fpr95": fpr95}


def open_set_metrics(rows, unknown_label):
    unknown = [row for row in rows if row["label"] == unknown_label]
    known = [row for row in rows if row["label"] != unknown_label]
    if not unknown or not known:
        return {"unknown_count": len(unknown), "known_count": len(known),
                "auroc_unknown": None, "aupr_out": None, "fpr95": None, "auoscr": None,
                "unavailable_reason": "both known and unknown targets are required"}
    def unknown_score(row):
        if "unknown_score" in row:
            score = float(row["unknown_score"])
        elif unknown_label in row["probabilities"]:
            score = row["probabilities"][unknown_label]
        else:
            raise ValueError(f"{row['id']}: no {unknown_label!r} probability or unknown_score")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{row['id']}: unknown score must be finite and in [0,1]")
        return score
    ranking = binary_ranking([unknown_score(row) for row in unknown],
                             [unknown_score(row) for row in known])
    unknown_scores_all = [(unknown_score(row), row["label"] == unknown_label) for row in rows]
    predicted_unknown = [(score >= 0.5, positive) for score, positive in unknown_scores_all]
    tp = sum(predicted and positive for predicted, positive in predicted_unknown)
    fp = sum(predicted and not positive for predicted, positive in predicted_unknown)
    fn = sum(not predicted and positive for predicted, positive in predicted_unknown)
    unknown_precision = tp / (tp + fp) if tp + fp else 0.0
    unknown_recall = tp / (tp + fn) if tp + fn else 0.0
    unknown_f1 = (2 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall)
                  if unknown_precision + unknown_recall else 0.0)
    # OSCR: known-class correct classification rate versus unknown false-positive rate.
    known_scored = []
    for row in known:
        probs = {key: value for key, value in row["probabilities"].items() if key != unknown_label}
        if not probs:
            raise ValueError(f"{row['id']}: no known-class probabilities for OSCR")
        pred = max(probs, key=probs.get)
        known_scored.append((max(probs.values()), float(pred == row["label"])))
    unknown_scored = [(max(value for key, value in row["probabilities"].items() if key != unknown_label), False, False)
                      for row in unknown]
    combined = sorted([(score, True, bool(correct)) for score, correct in known_scored] + unknown_scored,
                      key=lambda item: item[0], reverse=True)
    points = [(0.0, 0.0)]
    correct_known = accepted_unknown = 0
    index = 0
    while index < len(combined):
        threshold = combined[index][0]
        while index < len(combined) and combined[index][0] == threshold:
            _, is_known, correct = combined[index]
            if is_known:
                correct_known += correct
            else:
                accepted_unknown += 1
            index += 1
        points.append((accepted_unknown / len(unknown), correct_known / len(known)))
    auoscr = sum((x2 - x1) * (y1 + y2) / 2
                 for (x1, y1), (x2, y2) in zip(points, points[1:]))
    known_accuracy = sum(correct for _, correct in known_scored) / len(known_scored)
    return {"known_count": len(known), "unknown_count": len(unknown),
            "known_accuracy_without_reject": known_accuracy,
            "unknown_precision_at_0_5": unknown_precision,
            "unknown_recall_at_0_5": unknown_recall,
            "unknown_f1_at_0_5": unknown_f1,
            "unknown_threshold": 0.5,
            "auroc_unknown": ranking["auroc"], "aupr_out": ranking["aupr_out"],
            "fpr95": ranking["fpr95"], "auoscr": auoscr}


def summarize(rows, bins, unknown_label):
    n = len(rows)
    correct = nll = brier = 0.0
    labels = sorted({row["label"] for row in rows} | {key for row in rows for key in row["probabilities"]})
    class_counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}
    reciprocal_ranks = []
    recall_at_k = {3: 0, 5: 0}
    for row in rows:
        probs, label = row["probabilities"], row["label"]
        pred = max(probs, key=probs.get)
        correct += pred == label
        nll -= math.log(max(probs[label], 1e-15))
        brier += sum((value - float(key == label)) ** 2 for key, value in probs.items())
        class_counts[label]["support"] += 1
        class_counts[label]["tp"] += pred == label
        if pred != label:
            class_counts[pred]["fp"] += 1
            class_counts[label]["fn"] += 1
        ranked = sorted(probs, key=probs.get, reverse=True)
        rank = ranked.index(label) + 1
        reciprocal_ranks.append(1 / rank)
        for k in recall_at_k:
            recall_at_k[k] += label in ranked[:k]
    ece, reliability = ece_and_reliability(rows, bins)
    ordered = sorted(rows, key=lambda row: max(row["probabilities"].values()), reverse=True)
    cumulative_errors = 0
    aurc = 0.0
    for index, row in enumerate(ordered, 1):
        cumulative_errors += max(row["probabilities"], key=row["probabilities"].get) != row["label"]
        aurc += cumulative_errors / index
    per_class = {}
    class_f1 = []
    micro_tp = micro_fp = micro_fn = 0
    for label, counts in class_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": counts["support"], "precision": precision, "recall": recall, "f1": f1}
        class_f1.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0)
    return {"n": n, "accuracy": correct / n, "nll": nll / n, "brier": brier / n,
            "macro_f1": sum(class_f1) / len(class_f1), "micro_f1": micro_f1,
            "per_class": per_class, "mrr": sum(reciprocal_ranks) / n,
            "recall_at_k": {str(k): recall_at_k[k] / n for k in recall_at_k},
            "ece": ece, "ece_bins": bins, "reliability": reliability,
            "aurc": aurc / n,
            "coverage_at_1pct_risk": coverage_at_risk(rows, 0.01),
            "risk_at_80pct_coverage": risk_coverage(rows, 0.80),
            "risk_at_90pct_coverage": risk_coverage(rows, 0.90),
            "open_set": open_set_metrics(rows, unknown_label)}


def field_value(row, dotted_name):
    value = row
    for part in dotted_name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--unknown-label", default="UNKNOWN")
    parser.add_argument("--group-by", default="", help="optional top-level row field or metadata field")
    parser.add_argument("--output", type=Path, help="write JSON output instead of stdout")
    args = parser.parse_args()
    rows = load_rows(args.predictions, args.bins)
    result = {"metrics": summarize(rows, args.bins, args.unknown_label)}
    if args.group_by:
        grouped = defaultdict(list)
        for row in rows:
            value = field_value(row, args.group_by)
            grouped[str(value if value is not None else "<missing>")].append(row)
        result["groups"] = {name: summarize(group_rows, args.bins, args.unknown_label)
                            for name, group_rows in sorted(grouped.items())}
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
