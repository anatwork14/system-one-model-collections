"""Supervised objectives for schema-bounded Clef-style decisions."""

import torch
import torch.nn.functional as F


def question_loss(
    logits: torch.Tensor,
    target_index: int,
    label_smoothing: float = 0.05,
    brier_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined loss, smoothed CE, and multiclass Brier loss."""
    target = torch.tensor([target_index], device=logits.device, dtype=torch.long)
    ce = F.cross_entropy(logits.unsqueeze(0), target, label_smoothing=label_smoothing)
    probabilities = logits.float().softmax(dim=-1)
    one_hot = F.one_hot(target, num_classes=logits.numel()).squeeze(0).float()
    brier = (probabilities - one_hot).square().sum()
    return ce + brier_weight * brier, ce.detach(), brier.detach()


def option_target(question: dict, option_ids: tuple[str, ...]) -> int:
    target = question["target"]
    if question["type"] == "noul":
        target_id = "true" if bool(target) else "false"
    elif question["type"] == "score":
        target_id = str(target)
    else:
        target_id = str(target)
    try:
        return option_ids.index(target_id)
    except ValueError as error:
        raise ValueError(f"target {target_id!r} missing from schema options {option_ids}") from error
