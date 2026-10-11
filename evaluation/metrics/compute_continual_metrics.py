#!/usr/bin/env python3
"""Compute continual/open-world learning metrics from an accuracy matrix.

Input JSON: {"task_order": [...], "scores": [{"after_task": "pretraining"|task,
"task": task, "score": 0..1}, ...], optional "joint_reference" and "baseline"
maps keyed by task. Scores are measured on every task after every training stage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def compute(data):
    tasks = data.get("task_order")
    if not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("task_order must be a nonempty list of unique task names")
    stages = ["pretraining", *tasks]
    matrix = {}
    for row in data.get("scores", []):
        stage, task = row.get("after_task"), row.get("task")
        score = row.get("score")
        if stage not in stages or task not in tasks:
            raise ValueError(f"invalid stage/task pair: {stage!r}/{task!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"score for {stage}/{task} must be finite and in [0,1]")
        key = (stages.index(stage), task)
        if key in matrix:
            raise ValueError(f"duplicate score for {stage}/{task}")
        matrix[key] = float(score)
    missing = [(stage, task) for stage in stages for task in tasks if (stages.index(stage), task) not in matrix]
    if missing:
        raise ValueError(f"incomplete continual accuracy matrix; missing {missing[:8]}")
    final_stage = len(stages) - 1
    final = {task: matrix[(final_stage, task)] for task in tasks}
    diagonal = {task: matrix[(index + 1, task)] for index, task in enumerate(tasks)}
    forgetting = {}
    for index, task in enumerate(tasks[:-1]):
        best = max(matrix[(stage_index, task)] for stage_index in range(index + 1, len(stages)))
        forgetting[task] = best - final[task]
    bwt = {task: final[task] - diagonal[task] for task in tasks[:-1]}
    baseline = data.get("baseline")
    fwt = None
    if baseline is not None:
        if set(baseline) != set(tasks):
            raise ValueError("baseline must provide one score per task")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
               for value in baseline.values()):
            raise ValueError("baseline scores must be finite and in [0,1]")
        fwt = {task: matrix[(index, task)] - baseline[task] for index, task in enumerate(tasks)}
    joint = data.get("joint_reference")
    intransigence = None
    if joint is not None:
        if set(joint) != set(tasks):
            raise ValueError("joint_reference must provide one score per task")
        intransigence = {task: joint[task] - final[task] for task in tasks}
    average_accuracy = sum(final.values()) / len(tasks)
    average_incremental = sum(diagonal.values()) / len(tasks)
    average_forgetting = sum(forgetting.values()) / len(forgetting) if forgetting else 0.0
    backward_transfer = sum(bwt.values()) / len(bwt) if bwt else 0.0
    forward_transfer = sum(fwt.values()) / len(fwt) if fwt else None
    intransigence_value = sum(intransigence.values()) / len(intransigence) if intransigence else None
    return {
        "AA": average_accuracy,
        "AIA": average_incremental,
        "average_forgetting": average_forgetting,
        "BWT": backward_transfer,
        "FWT": forward_transfer,
        "intransigence": intransigence_value,
        "average_accuracy_final": average_accuracy,
        "average_incremental_accuracy": average_incremental,
        "forgetting_by_task": forgetting,
        "backward_transfer": backward_transfer,
        "backward_transfer_by_task": bwt,
        "forward_transfer": forward_transfer,
        "forward_transfer_by_task": fwt,
        "intransigence_by_task": intransigence,
        "final_accuracy_by_task": final,
        "incremental_accuracy_by_task": diagonal,
        "metric_definitions": {
            "AA": "mean final-stage score over all tasks",
            "AIA": "mean score on each task immediately after its training stage",
            "forgetting": "best post-learning score minus final score, excluding the last task",
            "BWT": "final score minus score immediately after learning, averaged over all but last task",
            "FWT": "score before a task is learned minus supplied baseline score; requires baseline map",
            "intransigence": "joint-reference score minus continual final score; requires joint reference map",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compute(json.loads(args.matrix.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
