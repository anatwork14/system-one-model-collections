#!/usr/bin/env python3
"""Fit lightweight state/action projections over frozen Qwen last-token vectors."""
import argparse, json
from pathlib import Path
import torch
import yaml
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class PairHeads(nn.Module):
    def __init__(self, hidden, dim):
        super().__init__()
        self.state = nn.Linear(hidden, dim, bias=False)
        self.action = nn.Linear(hidden, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
    def scores(self, state, actions):
        s = nn.functional.normalize(self.state(state), dim=-1)
        a = nn.functional.normalize(self.action(actions), dim=-1)
        return self.logit_scale.exp().clamp(max=100) * (a @ s)

def read_rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

def render(tok, role, content):
    return tok.apply_chat_template([{"role": role, "content": content}], tokenize=True,
                                   add_generation_prompt=(role == "user"), return_tensors="pt",
                                   enable_thinking=False)

@torch.inference_mode()
def vector(model, tok, text, device):
    ids = render(tok, "user", text).to(device)
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=True)
    return out.hidden_states[-1][0, -1].float().cpu()

def cache_vectors(rows, model, tok, device):
    cached = []
    for row in rows:
        state = vector(model, tok, "State:\n" + row["state"] + "\n\nQuestion:\n" + row["instruction"], device)
        acts = [vector(model, tok, "Candidate answer/action:\n" + c["id"] + ": " + c["text"], device) for c in row["candidates"]]
        cached.append((state, torch.stack(acts), [c["id"] for c in row["candidates"]].index(row["label"]), row))
    return cached

def evaluate(heads, rows, device):
    correct = total = 0
    with torch.no_grad():
        for state, actions, label, _ in rows:
            scores = heads.scores(state.to(device), actions.to(device))
            correct += int(scores.argmax().item() == label); total += 1
    return {"correct": correct, "count": total, "accuracy": correct / max(total, 1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if not cfg.get("backbone_frozen") or cfg.get("objective") != "candidate_set_infonce":
        raise SystemExit("Config must declare a frozen backbone and candidate_set_infonce objective")
    torch.manual_seed(int(cfg.get("seed", 17)))
    train_rows, val_rows = read_rows(a.train), read_rows(a.validation)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map=a.device).eval()
    for param in model.parameters(): param.requires_grad_(False)
    hidden = model.config.get_text_config().hidden_size
    train_vecs = cache_vectors(train_rows, model, tok, a.device)
    val_vecs = cache_vectors(val_rows, model, tok, a.device)
    heads = PairHeads(hidden, a.projection_dim).to(a.device)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.learning_rate)
    for epoch in range(a.epochs):
        order = torch.randperm(len(train_vecs)).tolist()
        total_loss = 0.0
        for i in order:
            state, actions, label, _ = train_vecs[i]
            scores = heads.scores(state.to(a.device), actions.to(a.device)).unsqueeze(0)
            loss = nn.functional.cross_entropy(scores, torch.tensor([label], device=a.device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total_loss += loss.item()
        print(json.dumps({"epoch": epoch + 1, "loss": total_loss / max(len(order), 1), "validation": evaluate(heads, val_vecs, a.device)}), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save(heads.state_dict(), a.output / "heads.pt")
    tok.save_pretrained(a.output)
    (a.output / "config.json").write_text(json.dumps({"base": str(a.base), "hidden_size": hidden,
        "projection_dim": a.projection_dim, "epochs": a.epochs, "learning_rate": a.learning_rate,
        "objective": "candidate-set InfoNCE via cross-entropy over state/action similarities",
        "backbone_frozen": True, "recipe": cfg}, indent=2) + "\n")
    (a.output / "final_eval.json").write_text(json.dumps(evaluate(heads, val_vecs, a.device), indent=2) + "\n")

if __name__ == "__main__":
    main()
