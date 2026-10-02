# SemIf / OpenJev

## 1. Status

Released implementation; frozen base-model baseline

## 2. Family

A — direct option-token scoring

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class; multimodal checkpoint  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

Frozen direct typed-option logits. It reproduces the semantic-decision interface and does not claim to reproduce Jev's undisclosed architecture.

## 5. Architecture

```text
state + question + typed options → prompt → one model forward → allowed option logits → softmax
```

## 6. Input

JSONL decision with state, question, and options/criteria.

## 7. Output

Probability per typed option; serialized response includes scores and timing.

## 8. Autoregressive?

No answer-token generation in direct mode.

## 9. Decision probability

Next-token vocabulary logits gathered at the decision position for declared option tokens, then normalized over the candidate set.

## 10. Training

None.

## 11. Trainable modules

None (0 trainable parameters).

## 12. Frozen modules

Entire Qwen3.5-4B checkpoint.

## 13. Training objective

None.

## 14. Training data

None.

## 15. Training configuration

No training configuration.

## 16. Published artifacts

Base tokenizer/model only, shared via symlink. Source examples/fixtures are under upstream.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

See scripts/smoke.sh; CLI: semif-score --mode direct --model ../../shared/models/qwen3.5-4b --input upstream/examples/decisions.jsonl --output results/smoke.jsonl (adjust relative path from method root as shown by the script).

## 19. Expected resources

The BF16 checkpoint is roughly 9 GB on disk; plan for about 10–14 GB GPU memory at short context, with exact peak depending on attention/KV buffers. This is an estimate, not a local measurement.

## 20. Known limitations

Depends on prompt and tokenizer option-token mapping; softmax is not calibration; no generated reasoning; candidate count/context constraints follow upstream implementation.

## 21. Why it matters scientifically

Can frozen pretrained option logits provide a useful zero-training System-One baseline?

## 22. Source provenance

Repository: [https://github.com/iwillcodeu/openjev](https://github.com/iwillcodeu/openjev)  
Requested ref: `master`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `none`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
