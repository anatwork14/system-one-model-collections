# CLM-style-4B reproduction

## 1. Status

Proposed controlled reproduction; not trained

## 2. Family

E — contrastive state/action representations

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

Freeze Qwen and train state/action projection heads so compatible action descriptions can be scored by similarity.

## 5. Architecture

```text
state → frozen Qwen → state projection z_s; candidate action → frozen Qwen → action projection z_a; dot products → softmax
```

## 6. Input

One state and candidate action/class descriptions.

## 7. Output

Similarity and normalized candidate probability.

## 8. Autoregressive?

No autoregressive answer generation; encoder-style forward passes for state and actions.

## 9. Decision probability

Softmax of state/action similarity scores divided by learned or fixed temperature.

## 10. Training

InfoNCE contrastive training on controlled Nimble-derived pairs.

## 11. Trainable modules

State and action projections; optional scalar temperature.

## 12. Frozen modules

Entire Qwen3.5-4B backbone.

## 13. Training objective

Batch InfoNCE: -log exp(sim(z_s,z_pos)/τ) divided by the sum over positive and negative action candidates.

## 14. Training data

Derive positive state→correct-candidate pairs from the same 2,676 Nimble train examples; other allowed candidates become hard negatives. Do not use the 324 holdout for training. Original labels are synthetic/model-checked.

## 15. Training configuration

Proposed: 20 epochs, AdamW LR 1e-3, projection dimension 512, seed 17, one request per update; frozen backbone vectors are cached before head optimization. The 2048-token limit applies to each state or candidate prompt. This small-data recipe is a reproduction design, not official CLM training config.

## 16. Published artifacts

No Qwen3.5-4B published checkpoint. CLM upstream implementation is a reference; local projection weights and tokenizer are saved under `weights/clm-heads/`.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

After training: `python scripts/infer.py --config configs/clm.yaml --input ../10_nimble_4b/data/processed/eval_pairs.jsonl --output results/eval.jsonl`. `scripts/eval.sh` also computes the frozen-holdout metrics. Training launcher is `scripts/train.sh`; not executed here.

## 19. Expected resources

The frozen 4B BF16 encoder is about 9 GB of weights; plan for roughly 10–14 GB at batch 1 plus activations, depending on pooling and sequence length. Estimate only.

## 20. Known limitations

A frozen pretrained base may not produce aligned state/action vectors without enough contrastive data; pair construction and negatives strongly affect results; 2,676 examples are small.

## 21. Why it matters scientifically

Can a frozen encoder with lightweight contrastive heads support dynamic candidate descriptions and novel classes?

## 22. Source provenance

Repository: [https://github.com/Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `controlled Nimble-derived pairs`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
