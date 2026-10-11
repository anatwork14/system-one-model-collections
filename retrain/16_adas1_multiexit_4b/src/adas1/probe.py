"""Small, identical typed-choice probes used only for the P0 layer study."""

import torch
from torch import nn


class LayerProbe(nn.Module):
    def __init__(self, hidden_size: int = 2560, projection_dim: int = 256,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.project = nn.Linear(hidden_size, projection_dim)
        self.scorer = nn.Sequential(
            nn.Linear(projection_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, question: torch.Tensor, global_state: torch.Tensor,
                options: torch.Tensor) -> torch.Tensor:
        """Return one schema-option logit per option."""
        question = self.project(self.norm(question))
        global_state = self.project(self.norm(global_state))
        options = self.project(self.norm(options))
        query = question.expand_as(options)
        state = global_state.expand_as(options)
        features = torch.cat(
            (query, options, query * options, torch.abs(query - options), state), dim=-1
        )
        return self.scorer(features).squeeze(-1)


def objective(logits: torch.Tensor, target: int, brier_weight: float = 0.1) -> torch.Tensor:
    target_tensor = torch.tensor([target], dtype=torch.long, device=logits.device)
    probabilities = logits.softmax(dim=-1)
    one_hot = torch.nn.functional.one_hot(target_tensor[0], num_classes=logits.numel()).to(logits.dtype)
    return torch.nn.functional.cross_entropy(logits.unsqueeze(0), target_tensor) + brier_weight * (
        probabilities - one_hot
    ).square().sum()
