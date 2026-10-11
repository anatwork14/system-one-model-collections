"""Typed supervised and distillation losses for M16 adaptive exits."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from collections.abc import Callable, Mapping, Sequence
from torch import nn

QUESTION_TYPES = {"noul", "choice", "score"}


class ExitHeadBank(nn.Module):
    """Run one schema head per selected decoder depth over matching hidden states."""

    def __init__(self, layers: Sequence[int], head_factory: Callable[[], nn.Module]) -> None:
        super().__init__()
        normalized = tuple(layers)
        if not normalized or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 1
                                 for layer in normalized):
            raise ValueError("layers must be a non-empty sequence of positive integers")
        if tuple(sorted(set(normalized))) != normalized:
            raise ValueError("layers must be unique and increasing")
        self.layers = normalized
        self.heads = nn.ModuleDict({str(layer): head_factory() for layer in normalized})

    def initialize_from(self, teacher_state: Mapping[str, torch.Tensor]) -> None:
        """Initialize each student exit from an already trained full-depth head."""
        for layer in self.layers:
            self.heads[str(layer)].load_state_dict(teacher_state)

    def forward(
        self,
        hidden_states: Mapping[int, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records,
        output_embedding_weight: torch.Tensor,
    ) -> dict[int, object]:
        missing = [layer for layer in self.layers if layer not in hidden_states]
        if missing:
            raise ValueError(f"hidden states are missing configured exit layers: {missing}")
        return {
            layer: self.heads[str(layer)](
                hidden_states[layer], input_ids, attention_mask, records, output_embedding_weight
            )
            for layer in self.layers
        }


def _validate_logits(logits: torch.Tensor, name: str) -> None:
    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two options")
    if not torch.isfinite(logits).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_target(target: int, option_count: int) -> None:
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < option_count:
        raise ValueError(f"target must be an integer index in [0, {option_count})")


def supervised_decision_loss(
    logits: torch.Tensor,
    target: int,
    brier_weight: float = 0.1,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Return cross-entropy plus multiclass Brier loss for one typed question."""
    _validate_logits(logits, "logits")
    _validate_target(target, logits.numel())
    if brier_weight < 0:
        raise ValueError("brier_weight must be nonnegative")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
    probabilities = logits.softmax(dim=-1)
    one_hot = F.one_hot(target_tensor[0], num_classes=logits.numel()).to(logits.dtype)
    return F.cross_entropy(logits.unsqueeze(0), target_tensor, label_smoothing=label_smoothing) + brier_weight * (
        probabilities - one_hot
    ).square().sum()


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student); teacher gradients are stopped."""
    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher must have the same option count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_probabilities = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probabilities = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(
        student_log_probabilities.unsqueeze(0),
        teacher_probabilities.unsqueeze(0),
        reduction="batchmean",
    ) * (temperature**2)


def ordinal_rps_loss(logits: torch.Tensor, target: int) -> torch.Tensor:
    """Ranked probability score for ordered score options, averaged over cut points."""
    _validate_logits(logits, "logits")
    _validate_target(target, logits.numel())
    probabilities = logits.softmax(dim=-1)
    predicted_cumulative = probabilities.cumsum(dim=-1)[:-1]
    boundaries = torch.arange(logits.numel() - 1, device=logits.device)
    target_cumulative = (boundaries >= target).to(dtype=probabilities.dtype)
    return (predicted_cumulative - target_cumulative).square().mean()


def multiexit_question_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: int,
    question_type: str,
    *,
    brier_weight: float = 0.1,
    label_smoothing: float = 0.05,
    distill_weight: float = 0.5,
    temperature: float = 2.0,
    ordinal_rps_weight: float = 0.1,
) -> torch.Tensor:
    """Supervise one exit and distill the full-depth teacher for one question."""
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {sorted(QUESTION_TYPES)}")
    if distill_weight < 0 or ordinal_rps_weight < 0:
        raise ValueError("distillation and ordinal RPS weights must be nonnegative")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher must have the same option count")
    # Compute the combined loss in fp32 even when GPU heads emit bf16/fp16 logits.
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    loss = supervised_decision_loss(student_logits, target, brier_weight, label_smoothing)
    loss = loss + distill_weight * distillation_kl_loss(
        student_logits, teacher_logits, temperature
    )
    if question_type == "score" and ordinal_rps_weight:
        loss = loss + ordinal_rps_weight * ordinal_rps_loss(student_logits, target)
    return loss
