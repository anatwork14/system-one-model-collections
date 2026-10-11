import pytest
import torch

from rlcd_objective import rlcd_loss
from openjev_rlcd.estimators import rb_surrogate, single_surrogate


def test_stage1_has_pathwise_gradient_and_no_score_function_gradient():
    logits = torch.tensor([[0.5, -0.3], [-0.2, 0.7]], requires_grad=True)
    rationale_logp = torch.tensor([-0.4, -0.9], requires_grad=True)
    loss = rlcd_loss(logits.softmax(-1), rationale_logp, 0, stage="stage1")
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert rationale_logp.grad is None


def test_stage2_has_pathwise_and_score_function_gradients():
    logits = torch.tensor([[0.5, -0.3], [-0.2, 0.7]], requires_grad=True)
    rationale_logp = torch.tensor([-0.4, -0.9], requires_grad=True)
    reference_logp = torch.tensor([-0.7, -0.6])
    loss = rlcd_loss(logits.softmax(-1), rationale_logp, 0, stage="stage2",
                     sf_coef=0.3, kl_beta=0.04, reference_logp=reference_logp)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert rationale_logp.grad is not None and rationale_logp.grad.abs().sum() > 0
    assert reference_logp.grad is None


def test_stage2_requires_frozen_reference_log_probs():
    lp = torch.tensor([-0.4, -0.9], requires_grad=True)
    with pytest.raises(ValueError, match="detached/frozen"):
        rlcd_loss(torch.full((2, 2), 0.5), lp, 0, stage="stage2", reference_logp=lp)


def test_stage1_objective_is_invariant_to_reference_and_sf_coefficient():
    u = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    logp = torch.tensor([-0.8, -0.2], requires_grad=True)
    a = rlcd_loss(u, logp, 0, stage="stage1", sf_coef=0.0)
    b = rlcd_loss(u, logp, 0, stage="stage1", sf_coef=0.3)
    assert torch.equal(a, b)


def test_diversity_weight_must_be_zero_for_m17():
    with pytest.raises(ValueError, match="diversity_weight=0"):
        rlcd_loss(torch.full((2, 2), 0.5), torch.zeros(2), 0,
                  stage="stage1", diversity_weight=1.0)


def test_zero_variance_brier_gives_same_per_rationale_and_mixture_score():
    u = torch.tensor([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3], [0.7, 0.3]], dtype=torch.float64,
                     requires_grad=True)
    logp = torch.tensor([-0.2, -0.7, -0.5, -0.9], dtype=torch.float64)
    single = single_surrogate(u, logp, 0)
    mixture = rb_surrogate(u, logp, 0)
    assert torch.allclose(single, mixture, atol=1e-12, rtol=0)
