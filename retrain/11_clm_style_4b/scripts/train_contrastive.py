#!/usr/bin/env python3
"""Fit lightweight state/action projections over frozen Qwen last-token vectors."""
import argparse, json
import hashlib
from pathlib import Path
import math
import tempfile
import torch
import yaml
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

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
    if role != "user":
        raise ValueError("CLM reproduction encodes user prompts only")
    prompt = tok.apply_chat_template([{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]

@torch.inference_mode()
def cache_vectors(rows, model, tok, device, split, batch_size=8):
    # Batch the frozen-backbone pass: one full model call per prompt was taking
    # several hours with Qwen's reference kernels on this GPU image.
    tok.padding_side = "right"
    prompts, layouts = [], []
    for row in rows:
        start = len(prompts)
        contents = ["State:\n" + row["state"] + "\n\nQuestion:\n" + row["instruction"]]
        contents.extend("Candidate answer/action:\n" + c["id"] + ": " + c["text"] for c in row["candidates"])
        prompts.extend(tok.apply_chat_template([{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for content in contents)
        layouts.append((start, len(contents), row))

    vectors = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        if batch_index % 25 == 0:
            print(json.dumps({"stage": "cache_vectors", "split": split,
                              "completed_prompts": start, "total_prompts": len(prompts),
                              "batch_size": batch_size}), flush=True)
        encoded = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=2048)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        out = model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
        last = encoded["attention_mask"].sum(dim=1) - 1
        batch_vectors = out.hidden_states[-1][torch.arange(last.shape[0], device=device), last]
        vectors.extend(batch_vectors.float().cpu().unbind(0))
        del out, encoded, batch_vectors
    cached = []
    for start, count, row in layouts:
        state, actions = vectors[start], torch.stack(vectors[start + 1:start + count])
        label = [c["id"] for c in row["candidates"]].index(row["label"])
        cached.append((state, actions, label, row))
    print(json.dumps({"stage": "cache_vectors", "split": split,
                      "completed": len(rows), "total": len(rows),
                      "total_prompts": len(prompts), "total_batches": total_batches}), flush=True)
    return cached

def evaluate(heads, rows, device):
    correct = total = 0
    loss_total = 0.0
    with torch.no_grad():
        for state, actions, label, _ in rows:
            scores = heads.scores(state.to(device), actions.to(device))
            loss_total += float(nn.functional.cross_entropy(
                scores.unsqueeze(0), torch.tensor([label], device=device)))
            correct += int(scores.argmax().item() == label); total += 1
    return {"correct": correct, "count": total, "accuracy": correct / max(total, 1),
            "mean_loss": loss_total / max(total, 1)}


def save_state_dict(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--revision", help="Pinned Hugging Face model revision")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    if not cfg.get("backbone_frozen") or cfg.get("objective") != "candidate_set_infonce":
        raise SystemExit("Config must declare a frozen backbone and candidate_set_infonce objective")
    torch.manual_seed(int(cfg.get("seed", 17)))
    train_rows, val_rows = read_rows(a.train), read_rows(a.validation)
    split_manifest_path = Path(cfg.get("dataset_manifest", "data/manifests/inner_split.json"))
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Clean split manifest missing: {split_manifest_path}")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    for path, expected_rows in ((Path(a.train), len(train_rows)), (Path(a.validation), len(val_rows))):
        expected = split_manifest.get("files", {}).get(path.name)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or expected.get("rows") != expected_rows or expected.get("sha256") != actual_sha:
            raise ValueError(f"Input does not match the generated family-disjoint split manifest: {path}")
    train_ids = {row["id"] for row in train_rows}
    val_ids = {row["id"] for row in val_rows}
    train_families = {row.get("family") for row in train_rows}
    val_families = {row.get("family") for row in val_rows}
    if not train_rows or not val_rows or train_ids & val_ids or train_families & val_families:
        raise ValueError("Training and inner-dev rows must be nonempty and disjoint by id and source family")
    if any(family is None for family in train_families | val_families):
        raise ValueError("Every row must include its source family for leakage checks")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_dev_loss = math.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    print(json.dumps({"stage": "load_tokenizer", "base": str(a.base), "revision": a.revision}), flush=True)
    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    print(json.dumps({"stage": "load_backbone", "base": str(a.base), "revision": a.revision}), flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.base, revision=a.revision, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.device).eval()
    model.config.use_cache = False
    for param in model.parameters(): param.requires_grad_(False)
    hidden = model.config.get_text_config().hidden_size
    print(json.dumps({"stage": "backbone_ready", "hidden_size": hidden,
                      "train_rows": len(train_rows), "validation_rows": len(val_rows)}), flush=True)
    train_vecs = cache_vectors(train_rows, model, tok, a.device, "train")
    val_vecs = cache_vectors(val_rows, model, tok, a.device, "validation")
    heads = PairHeads(hidden, a.projection_dim).to(a.device)
    opt = torch.optim.AdamW(heads.parameters(), lr=a.learning_rate)
    for epoch in range(a.epochs):
        order = torch.randperm(len(train_vecs)).tolist()
        total_loss = 0.0
        for i in order:
            state, actions, label, _ = train_vecs[i]
            scores = heads.scores(state.to(a.device), actions.to(a.device)).unsqueeze(0)
            loss = nn.functional.cross_entropy(scores, torch.tensor([label], device=a.device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite contrastive loss at epoch {epoch + 1}, row {i}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total_loss += loss.item()
        dev_metrics = evaluate(heads, val_vecs, a.device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(len(order), 1), "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if dev_metrics["mean_loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["mean_loss"]
            best_epoch = epoch + 1
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()}
            save_state_dict(best_state, a.output / "heads.pt")
        else:
            bad_epochs += 1
        (a.output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if bad_epochs >= int(cfg.get("early_stopping_patience", 5)):
            break
    if best_state is None:
        raise RuntimeError("No finite inner-dev checkpoint was selected")
    heads.load_state_dict(best_state)
    tok.save_pretrained(a.output)
    (a.output / "config.json").write_text(json.dumps({"base": str(a.base), "hidden_size": hidden,
        "revision": a.revision, "projection_dim": a.projection_dim, "epochs_requested": a.epochs,
        "epochs_completed": len(history), "best_epoch": best_epoch, "best_dev_loss": best_dev_loss,
        "learning_rate": a.learning_rate,
        "objective": "candidate-set InfoNCE via cross-entropy over state/action similarities",
        "backbone_frozen": True, "selection_split": "inner_dev", "outer_eval_used": False,
        "train_families": sorted(train_families), "dev_families": sorted(val_families),
        "recipe": cfg}, indent=2) + "\n")
    (a.output / "dev_eval.json").write_text(json.dumps(evaluate(heads, val_vecs, a.device), indent=2) + "\n")

if __name__ == "__main__":
    main()
