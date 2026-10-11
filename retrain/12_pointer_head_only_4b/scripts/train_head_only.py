#!/usr/bin/env python3
"""Train only Kev's pointer head on frozen Qwen hidden states.

The data adapter and pointer architecture come from the pinned, read-only Kev checkout.
This is a small controlled ablation trainer; it does not modify upstream.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


def load_upstream(upstream: Path):
    sys.path.insert(0, str(upstream.resolve()))
    from kev.data import load_records, materialize
    from kev.model import DecisionModel, load_tokenizer, training_context, rows_of
    from kev.train import question_loss
    return load_records, materialize, DecisionModel, load_tokenizer, training_context, question_loss, rows_of


@torch.no_grad()
def extract_record_features(model, tok, rec, ctx, device, rows_of_fn):
    enc = model.encode(tok, rec, strict=True, max_state=ctx["max_state"], max_branch=ctx["max_branch"])
    S, Sp, brs = rows_of_fn(enc)
    rows = [(S + r["ids"], Sp + r["pos"]) for r in brs]
    readouts = [(len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for r in brs]
    hs = model._rows_hidden(rows)
    q_features = []
    for h, (d, oi), q in zip(hs, readouts, rec["questions"]):
        h_d = h[d].detach().cpu()
        h_o = h[torch.tensor(oi, device=device)].detach().cpu()
        q_features.append({"h_d": h_d, "h_o": h_o, "question": q})
    return q_features


def score_features(model_head, cached_records, device, question_loss_fn=None, train=False, optimizer=None):
    total_loss = 0.0
    correct = count = 0
    for q_features in cached_records:
        losses = []
        for item in q_features:
            h_d = item["h_d"].to(device)
            h_o = item["h_o"].to(device)
            question = item["question"]
            z = model_head(h_d, h_o).float()
            if not torch.isfinite(z).all():
                raise FloatingPointError("Non-finite pointer scores")
            loss_i = question_loss_fn(z, question, device, 0.0)
            if not torch.isfinite(loss_i):
                raise FloatingPointError("Non-finite pointer cross-entropy")
            losses.append(loss_i)
            correct += int(z.argmax().item() == int(question["label"]))
            count += 1
        loss = torch.stack(losses).mean()
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        total_loss += float(loss.detach())
    return {
        "mean_record_loss": total_loss / max(len(cached_records), 1),
        "question_accuracy": correct / max(count, 1),
        "questions": count,
    }


def save_head(model_head, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in model_head.state_dict().items()}, temporary)
    temporary.replace(path)


def save_history(history, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--base-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, default=Path("data/manifests/inner_split.json"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--max-state", type=int, default=2048)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    (load_records, materialize, DecisionModel, load_tokenizer,
     training_context, question_loss_fn, rows_of) = load_upstream(a.upstream)
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    train_rows = load_records(a.train)
    validation_rows = load_records(a.validation)
    split_manifest = json.loads(a.split_manifest.read_text(encoding="utf-8"))
    for path, expected_rows in ((a.train, len(train_rows)), (a.validation, len(validation_rows))):
        expected = split_manifest.get("files", {}).get(path.name)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or expected.get("rows") != expected_rows or expected.get("sha256") != actual_sha:
            raise ValueError(f"Input does not match the generated family-disjoint split manifest: {path}")
    train_ids = {row.get("_meta", {}).get("id") for row in train_rows}
    dev_ids = {row.get("_meta", {}).get("id") for row in validation_rows}
    train_families = {row.get("_meta", {}).get("source_family") for row in train_rows}
    dev_families = {row.get("_meta", {}).get("source_family") for row in validation_rows}
    if not train_rows or not validation_rows or None in train_ids | dev_ids or None in train_families | dev_families:
        raise ValueError("Training and inner-dev data need IDs and source_family metadata")
    if train_ids & dev_ids or train_families & dev_families:
        raise ValueError("Training and inner-dev records overlap by ID or source family")
    if a.epochs <= 0 or a.learning_rate <= 0 or a.patience <= 0:
        raise ValueError("epochs, learning rate, and patience must be positive")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(a.base, revision=a.base_revision)
    model = DecisionModel(a.base, tok, a.device, lora=None, revision=a.base_revision,
                          head_dim=a.head_dim, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.eval()
    for param in model.lm.parameters():
        param.requires_grad_(False)
    if any(param.requires_grad for param in model.lm.parameters()):
        raise RuntimeError("backbone is not fully frozen")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    ctx = training_context(a.max_state)
    print(json.dumps({"stage": "training_context", "max_state": ctx["max_state"],
                      "max_branch": ctx["max_branch"], "max_packed": ctx["max_packed"]}), flush=True)

    rng_state = random.getstate()

    print(json.dumps({"stage": "feature_extraction_start", "train_records": len(train_rows), "val_records": len(validation_rows)}), flush=True)
    train_cached = []
    for i, request in enumerate(train_rows):
        rec = materialize(request)
        train_cached.append(extract_record_features(model, tok, rec, ctx, device, rows_of))
        if (i + 1) % 500 == 0 or (i + 1) == len(train_rows):
            print(json.dumps({"stage": "feature_extraction_train", "done": i + 1, "total": len(train_rows)}), flush=True)

    val_cached = []
    for i, request in enumerate(validation_rows):
        rec = materialize(request)
        val_cached.append(extract_record_features(model, tok, rec, ctx, device, rows_of))
        if (i + 1) % 100 == 0 or (i + 1) == len(validation_rows):
            print(json.dumps({"stage": "feature_extraction_val", "done": i + 1, "total": len(validation_rows)}), flush=True)

    print(json.dumps({"stage": "feature_extraction_complete"}), flush=True)

    # Free the heavy backbone to reclaim memory and keep only the head on GPU
    del model.lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore RNG state so Epoch 1 shuffle precisely matches the seed baseline
    random.setstate(rng_state)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_head = None
    for epoch in range(a.epochs):
        random.shuffle(train_cached)
        model.head.train()
        train_metrics = score_features(model.head, train_cached, device, question_loss_fn, train=True, optimizer=optimizer)
        model.head.eval()
        with torch.no_grad():
            val_metrics = score_features(model.head, val_cached, device, question_loss_fn, train=False)
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)

        if val_metrics["mean_record_loss"] < best_val_loss:
            best_val_loss = val_metrics["mean_record_loss"]
            best_epoch = epoch + 1
            stale_epochs = 0
            best_head = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}
            save_head(model.head, a.output / "pointer_head_best.pt")
        else:
            stale_epochs += 1
        save_head(model.head, a.output / f"pointer_head_epoch_{epoch + 1:02d}.pt")
        save_history(history, a.output / "training_history.json")
        if stale_epochs >= a.patience:
            break

    if best_head is None:
        raise RuntimeError("No finite inner-dev checkpoint was selected")
    model.head.load_state_dict(best_head)
    save_head(model.head, a.output / "pointer_head.pt")

    (a.output / "config.json").write_text(json.dumps({
        "method": "pointer-head-only-4b", "base": a.base, "base_revision": a.base_revision,
        "head_dim": a.head_dim, "seed": a.seed, "epochs_requested": a.epochs,
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_inner_dev_loss": best_val_loss, "selection_metric": "mean_record_loss",
        "learning_rate": a.learning_rate, "weight_decay": a.weight_decay,
        "max_state": ctx["max_state"], "max_branch": ctx["max_branch"],
        "max_packed": ctx["max_packed"], "backbone_frozen": True,
        "feature_extraction_mode": "eval/no_grad; cache state and option readouts before head training",
        "data": "Nimble original training split, family-disjoint inner dev",
        "train_records": len(train_rows), "dev_records": len(validation_rows),
        "train_source_families": sorted(train_families), "dev_source_families": sorted(dev_families),
        "outer_eval_used_for_training_or_selection": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "training_finished", "epochs_completed": len(history),
                      "best_epoch": best_epoch, "best_inner_dev_loss": best_val_loss}), flush=True)


if __name__ == "__main__":
    main()
