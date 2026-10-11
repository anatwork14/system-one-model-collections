"""Training-time utilities for M16 teacher and multi-exit objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import torch

from .multiexit import multiexit_question_loss


def parse_exit_layers(value: str, num_hidden_layers: int = 32) -> tuple[int, ...]:
    """Parse a unique, increasing list of student exits strictly before full depth."""
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("at least one intermediate exit layer is required")
    try:
        layers = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("exit layers must be comma-separated integers") from exc
    if tuple(sorted(set(layers))) != layers:
        raise ValueError("exit layers must be unique and increasing")
    if num_hidden_layers < 2 or any(layer < 1 or layer >= num_hidden_layers for layer in layers):
        raise ValueError(f"student exit layers must be between 1 and {num_hidden_layers - 1}")
    return layers


def summarize_decisions(logits_list, targets: Sequence[int], bins: int = 15) -> dict[str, float | int]:
    """Compute accuracy, NLL, Brier, and top-label ECE for variable-size decisions."""
    import math

    if bins < 1:
        raise ValueError("bins must be positive")
    if len(logits_list) != len(targets) or not logits_list:
        raise ValueError("logits and targets must have the same nonzero count")
    rows = []
    for logits, target in zip(logits_list, targets):
        logits = logits.detach().float()
        if logits.ndim != 1 or logits.numel() < 2 or not torch.isfinite(logits).all():
            raise ValueError("each decision must contain finite logits for at least two options")
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < logits.numel():
            raise ValueError("target index is outside the available options")
        probabilities = logits.softmax(dim=-1)
        prediction = int(probabilities.argmax().item())
        confidence = float(probabilities[prediction].item())
        correct = int(prediction == target)
        nll = -math.log(max(float(probabilities[target].item()), torch.finfo(torch.float32).tiny))
        one_hot = torch.nn.functional.one_hot(torch.tensor(target), num_classes=logits.numel()).float()
        brier = float((probabilities - one_hot).square().sum().item())
        rows.append((confidence, correct, nll, brier))
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        members = [row for row in rows if low <= row[0] < high or (bin_index == bins - 1 and row[0] == 1.0)]
        if members:
            confidence = sum(row[0] for row in members) / len(members)
            accuracy = sum(row[1] for row in members) / len(members)
            ece += len(members) / len(rows) * abs(confidence - accuracy)
    return {
        "count": len(rows),
        "accuracy": sum(row[1] for row in rows) / len(rows),
        "mean_nll": sum(row[2] for row in rows) / len(rows),
        "mean_brier": sum(row[3] for row in rows) / len(rows),
        "ece": ece,
    }


def target_index(question: Mapping, option_ids: Sequence[str]) -> int:
    """Map a typed gold target to the exact option order emitted by the schema encoder."""
    question_type = question.get("type")
    target = question.get("target")
    if question_type == "noul":
        if not isinstance(target, bool):
            raise ValueError("noul targets must be boolean")
        target_id = "true" if target else "false"
    elif question_type in {"choice", "score"}:
        target_id = str(target)
    else:
        raise ValueError(f"unsupported question type: {question_type!r}")
    try:
        return tuple(option_ids).index(target_id)
    except ValueError as exc:
        raise ValueError(f"target option {target_id!r} is missing from {tuple(option_ids)!r}") from exc


def aggregate_multiexit_loss(
    outputs_by_layer: Mapping[int, Sequence[Sequence[torch.Tensor]]],
    targets: Sequence[Sequence[int]],
    question_types: Sequence[Sequence[str]],
    *,
    teacher_layer: int = 32,
    **loss_kwargs,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Average supervised/distillation loss across records, questions, and student exits.

    Each layer maps to the nested per-record/per-question logits returned by
    ``JointSchemaHead``. The teacher's logits are detached by the KL objective.
    """
    if teacher_layer not in outputs_by_layer:
        raise ValueError(f"teacher outputs are missing configured layer {teacher_layer}")
    teacher_outputs = outputs_by_layer[teacher_layer]
    record_count = len(targets)
    if len(question_types) != record_count or len(teacher_outputs) != record_count:
        raise ValueError("record counts must match across targets, types, and teacher outputs")
    student_layers = [layer for layer in outputs_by_layer if layer != teacher_layer]
    if not student_layers:
        raise ValueError("at least one student exit is required in addition to the teacher")

    per_exit: dict[int, torch.Tensor] = {}
    for layer in student_layers:
        student_outputs = outputs_by_layer[layer]
        if len(student_outputs) != record_count:
            raise ValueError(f"record counts must match for exit layer {layer}")
        question_losses: list[torch.Tensor] = []
        for record_index, (student_record, teacher_record, record_targets, record_types) in enumerate(
            zip(student_outputs, teacher_outputs, targets, question_types)
        ):
            expected_count = len(record_targets)
            if not (len(record_types) == len(student_record) == len(teacher_record) == expected_count):
                raise ValueError(
                    f"question counts must match for exit layer {layer}, record {record_index}"
                )
            for student_logits, teacher_logits, target, question_type in zip(
                student_record, teacher_record, record_targets, record_types
            ):
                question_losses.append(
                    multiexit_question_loss(
                        student_logits, teacher_logits, int(target), str(question_type), **loss_kwargs
                    )
                )
        if not question_losses:
            raise ValueError(f"exit layer {layer} has no questions to train on")
        per_exit[layer] = torch.stack(question_losses).mean()
    return torch.stack(list(per_exit.values())).mean(), per_exit


def load_controlled_splits(data_dir):
    """Load and validate only the pinned train/dev files; never inspect outer-test files."""
    import json
    from pathlib import Path

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "nimble-m16-p0-controlled-v1":
        raise ValueError("unexpected M16 controlled dataset manifest")
    if manifest.get("outer_324_eval_read") is not False:
        raise ValueError("controlled manifest must assert that outer test was not read")
    controlled_hashes = manifest.get("controlled_sha256")
    if not isinstance(controlled_hashes, dict):
        raise ValueError("controlled manifest must pin canonical train/dev sha256 values")
    # Intentionally name only the two allowed split files; do not glob this directory.
    split_rows = {}
    for split in ("train", "dev"):
        path = data_dir / f"{split}.jsonl"
        raw = path.read_bytes()
        expected_hash = controlled_hashes.get(split)
        actual_hash = hashlib.sha256(raw).hexdigest()
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError(f"controlled {split} sha256 mismatch")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        expected = manifest.get("counts", {}).get(split)
        if not rows or expected != len(rows):
            raise ValueError(f"{split} count does not match the controlled manifest")
        split_rows[split] = rows

    def identifiers(rows, key):
        return {str(row.get(key, "")) for row in rows}

    for key in ("id", "family", "source_id"):
        overlap = identifiers(split_rows["train"], key) & identifiers(split_rows["dev"], key)
        if overlap:
            raise ValueError(f"train/dev overlap by {key}: {sorted(overlap)[:3]}")
    return split_rows["train"], split_rows["dev"]
