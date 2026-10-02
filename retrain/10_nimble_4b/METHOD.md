# Nimble-4B reproduction

## 1. Status

Proposed reproduction; not trained

## 2. Family

B — candidate-only LM-logit CE

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

Adapt the published schema-aware candidate-logit training recipe from Qwen3.5-9B to 4B.

## 5. Architecture

```text
schema decision → Qwen + LoRA → allowed candidate logits only → candidate softmax
```

## 6. Input

Schema-aware record with state/context, field/question, and allowed choices.

## 7. Output

Candidate probability distribution.

## 8. Autoregressive?

No generated output at inference; candidate scoring from one forward pass.

## 9. Decision probability

Softmax over the allowed candidate-token logits.

## 10. Training

LoRA with candidate-only cross-entropy; one epoch.

## 11. Trainable modules

LoRA r=16; standard LM head is retained.

## 12. Frozen modules

Qwen weights outside LoRA.

## 13. Training objective

L = -log(exp(z_y) / sum_{c in C} exp(z_c)); disallowed vocabulary tokens do not compete in the decision loss.

## 14. Training data

2,676 train and 324 frozen evaluation examples. Ten domains and Choice/Noul/Score tasks. Labels are synthetic/model-checked and not human-reviewed. Preserve upstream license and provenance.

## 15. Training configuration

LR 5e-5; seed 17; batch 2; accumulation 4 (effective batch 8); max length 2048; one epoch; BF16; LoRA rank 16. Backbone size is the intended change from published 9B recipe.

## 16. Published artifacts

No published 4B checkpoint. Upstream Nimble trainer/data; local adapter output in weights/nimble-4b.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

After training: bash scripts/eval.sh. Train entry point: bash scripts/train.sh. No training is run by workspace setup.

## 19. Expected resources

Plan a 24 GB class GPU for BF16 LoRA with batch 2, accumulation 4, and length 2048; this is a conservative planning target, not a measured peak. Activation checkpointing/shorter microbatches may be needed. Disk includes the shared base and a small adapter.

## 20. Known limitations

Small synthetic/model-checked dataset; domain coverage and label quality constrain conclusions; frozen holdout must never tune hyperparameters; tokenizer export from 9B may need regeneration for 4B.

## 21. Why it matters scientifically

Does candidate-only decision CE provide an advantage over ordinary answer-token SFT on the same examples?

## 22. Source provenance

Repository: [https://github.com/bespokelabsai/nimble](https://github.com/bespokelabsai/nimble)  
Requested ref: `main`
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none; official checkpoint is Qwen3.5-9B based`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `2,676 train + 324 held-out`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
