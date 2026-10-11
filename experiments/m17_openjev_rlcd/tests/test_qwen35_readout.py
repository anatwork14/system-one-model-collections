import types

import pytest
import torch

from train_m17 import RationaleReader, discover_lora_targets


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == " yes":
            return [5]
        if text == " no":
            return [6]
        if text == "\nAnswer:":
            return [7, 8]
        return []


def test_label_tokens_are_exact_and_distinct():
    reader = RationaleReader(FakeTokenizer())
    assert reader.label_ids == [5, 6]
    assert len(reader.suffix) > 0


def test_label_tokenization_rejects_multi_token_labels():
    t = FakeTokenizer()
    t.encode = lambda text, add_special_tokens=False: [5, 9] if text == " yes" else [6]
    with pytest.raises(ValueError, match="exactly one token"):
        RationaleReader(t)


def test_targets_cover_hybrid_attention_and_delta_net():
    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = types.SimpleNamespace()
            self.add_module("q_proj", torch.nn.Linear(4, 4))
            self.add_module("in_proj_qkv", torch.nn.Linear(4, 4))
            self.add_module("out_proj", torch.nn.Linear(4, 4))

    assert set(discover_lora_targets(Graph())) == {"q_proj", "in_proj_qkv", "out_proj"}


def test_readout_probabilities_sum_to_one():
    z = torch.tensor([[2.0, -1.0], [-4.0, 3.0]])
    p = z.softmax(-1)
    assert torch.allclose(p.sum(-1), torch.ones(2))
