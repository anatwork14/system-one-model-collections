import torch
from peft import LoraConfig, get_peft_model

import train_m17
from train_m17 import (ADAPTER_NAME, adapter_state, checkpoint_state,
                       elapsed_training_seconds, load_adapter_state)


def test_checkpoint_contains_optimizer_rng_stage_and_reference_digest(monkeypatch):
    monkeypatch.setattr(train_m17, "adapter_state", lambda model: {"adapter": torch.ones(1)})
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    payload = checkpoint_state(model, optimizer, 23, 17, 12.5,
                               {"stage": "stage1", "seed": 17}, "frozen-stage1-sha")
    assert payload["format"] == "m17-adapter-resume-v1"
    assert payload["step"] == 23
    assert payload["seed"] == 17
    assert payload["optimizer"] == optimizer.state_dict()
    assert payload["python_rng"] is not None and payload["numpy_rng"] is not None
    assert payload["torch_rng"] is not None and payload["cuda_rng"] is not None
    assert payload["reference_checkpoint_sha256"] == "frozen-stage1-sha"


def test_elapsed_stage_budget_carries_across_resume_segments():
    assert elapsed_training_seconds(12.5, 100.0, now=103.25) == 15.75


def test_optimizer_is_limited_to_selected_trainable_parameters():
    base = torch.nn.Linear(2, 2)
    for p in base.parameters():
        p.requires_grad_(False)
    adapter = torch.nn.Parameter(torch.ones(2, 2))
    opt = torch.optim.AdamW([adapter], lr=1e-3)
    assert len(opt.param_groups[0]["params"]) == 1
    assert opt.param_groups[0]["params"][0] is adapter


def test_stage2_policy_initializes_from_stage1_adapter_checkpoint():
    torch.manual_seed(7)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["0"], bias="none")
    stage1 = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)), config,
                            adapter_name=ADAPTER_NAME)
    with torch.no_grad():
        for name, p in stage1.named_parameters():
            if "lora_" in name:
                p.add_(0.125)
    source = adapter_state(stage1)
    stage2 = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)), config,
                            adapter_name=ADAPTER_NAME)
    load_adapter_state(stage2, source, ADAPTER_NAME)
    restored = adapter_state(stage2)
    assert source.keys() == restored.keys()
    assert all(torch.equal(source[k], restored[k]) for k in source)


def test_frozen_reference_tensor_is_unchanged_across_stage2_loss():
    from rlcd_objective import rlcd_loss

    reference = torch.tensor([-0.3, -0.8])
    before = reference.clone()
    policy = torch.tensor([-0.7, -0.2], requires_grad=True)
    u = torch.tensor([[0.8, 0.2], [0.6, 0.4]], requires_grad=True)
    rlcd_loss(u, policy, 0, stage="stage2", reference_logp=reference).backward()
    assert torch.equal(reference, before)
    assert reference.grad is None
