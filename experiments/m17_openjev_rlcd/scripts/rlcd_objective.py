"""Small semantic boundary for the Stage-1/Stage-2 estimator paths."""
from __future__ import annotations

import torch

from openjev_rlcd.estimators import lambda_surrogate


def rlcd_loss(u: torch.Tensor, rationale_logp: torch.Tensor, y: int, *, stage: str,
              sf_coef: float = 0.3, kl_beta: float = 0.04,
              reference_logp: torch.Tensor | None = None,
              diversity_weight: float = 0.0) -> torch.Tensor:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unknown RLCD stage: {stage}")
    if diversity_weight != 0.0:
        raise ValueError("M17 per-rationale objective requires diversity_weight=0")
    if stage == "stage1":
        # The reference code computes shaping, but its contribution is score-function-only.
        # Detaching here preserves the original Stage-1 pathwise-only behavior.
        return lambda_surrogate(u, rationale_logp.detach(), y, 0.0)
    if reference_logp is None:
        raise ValueError("Stage 2 requires log probabilities from the frozen Stage-1 reference")
    if reference_logp.requires_grad:
        raise ValueError("Stage-2 reference log probabilities must be detached/frozen")
    if reference_logp.shape != rationale_logp.shape:
        raise ValueError("policy/reference rationale log probabilities must have matching shapes")
    m = u.shape[0]
    extra = -(kl_beta / m) * (rationale_logp.detach() - reference_logp)
    score_logp = rationale_logp * sf_coef
    return lambda_surrogate(u, score_logp, y, 0.0, extra_local=extra)
