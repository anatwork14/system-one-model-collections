#!/usr/bin/env python3
"""Train only Kev's pointer head on frozen Qwen hidden states.

The data adapter and pointer architecture come from the pinned, read-only Kev checkout.
This is a small controlled ablation trainer; it does not modify upstream.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch


def load_upstream(upstream: Path):
    sys.path.insert(0, str(upstream.resolve()))
    from kev.data import load_records, materialize
    from kev.model import DecisionModel, load_tokenizer, training_context
    from kev.train import question_loss
    return load_records, materialize, DecisionModel, load_tokenizer, training_context, question_loss


def score(model, tok, records, device, ctx, question_loss_fn=None, train=False, optimizer=None):
    total_loss = 0.0
    correct = count = 0
    for request in records:
        rec = materialize(request)
        enc = model.encode(tok, rec, strict=True, max_state=ctx["max_state"], max_branch=ctx["max_branch"])
        logits = model(enc)
        losses = []
        for z, question in zip(logits, rec["questions"]):
            z = z.float()
            losses.append(question_loss_fn(z, question, device, 0.0))
            correct += int(z.argmax().item() == int(question["label"]))
            count += 1
        loss = torch.stack(losses).mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.detach())
    return {"mean_record_loss": total_loss / max(len(records), 1),
            "question_accuracy": correct / max(count, 1), "questions": count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--base-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    (load_records, materialize, DecisionModel, load_tokenizer,
     training_context, question_loss_fn) = load_upstream(a.upstream)
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    tok = load_tokenizer(a.base, revision=a.base_revision)
    model = DecisionModel(a.base, tok, a.device, lora=None, revision=a.base_revision,
                          head_dim=a.head_dim, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    for param in model.lm.parameters():
        param.requires_grad_(False)
    if any(param.requires_grad for param in model.lm.parameters()):
        raise RuntimeError("backbone is not fully frozen")
    train_rows = load_records(a.train)
    validation_rows = load_records(a.validation)
    if len(train_rows) != 2676 or len(validation_rows) != 324:
        raise ValueError(f"Expected frozen Nimble split sizes 2676/324; got {len(train_rows)}/{len(validation_rows)}")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    ctx = training_context()
    history = []
    for epoch in range(a.epochs):
        random.shuffle(train_rows)
        model.train(); model.lm.eval()
        train_metrics = score(model, tok, train_rows, device, ctx, question_loss_fn, True, optimizer)
        model.eval()
        val_metrics = score(model, tok, validation_rows, device, ctx, question_loss_fn)
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save({k: v.detach().cpu() for k, v in model.head.state_dict().items()}, a.output / "pointer_head.pt")
    (a.output / "config.json").write_text(json.dumps({
        "method": "pointer-head-only-4b", "base": a.base, "base_revision": a.base_revision,
        "head_dim": a.head_dim, "seed": a.seed, "epochs": a.epochs,
        "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
        "backbone_frozen": True, "data": "Nimble original-2676 controlled split",
    }, indent=2) + "\n", encoding="utf-8")
    (a.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
