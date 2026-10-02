# Simple Jev

## 1. Status

Released implementation; zero-shot inference by default

## 2. Family

A — direct option-token scoring / classifier serving

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class; multimodal checkpoint  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

Prompt compiler and classifier scorer expose typed Choice, Score, and Boolean decisions over compatible Hugging Face models. Server constructs structured output from next-token logits; model does not generate JSON.

## 5. Architecture

```text
typed request → prompt compiler → HF model next-token logits → response scorer → typed probabilities
```

## 6. Input

Classifier API request containing state and named typed questions with instructions and candidate values.

## 7. Output

Structured typed scores/probabilities; server emits JSON response.

## 8. Autoregressive?

No structured answer generation; reads next-token logits.

## 9. Decision probability

Scorer maps allowed answer/option token logits into the requested typed distribution.

## 10. Training

None for default serving. RFDT training workflow is available upstream and is separate from baseline inference.

## 11. Trainable modules

None at inference.

## 12. Frozen modules

Selected HF model at inference.

## 13. Training objective

None for default inference; optional RFDT workflow has its own upstream objective.

## 14. Training data

None required for default serving.

## 15. Training configuration

Server/runtime configuration only; use pinned model ID and the upstream Qwen dense 4B prompt policy if supported by checked-out revision.

## 16. Published artifacts

Shared base model. No method-specific checkpoint for baseline.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

See scripts/smoke.sh and upstream README; install the upstream HF server package and start with --model ../../shared/models/qwen3.5-4b.

## 19. Expected resources

Plan for the roughly 9 GB BF16 model plus server/batching buffers (about 10–14 GB for a short single-request smoke, estimate only). Larger batch/context limits increase memory.

## 20. Known limitations

Compatibility depends on prompt policy, tokenizer, and allowed answer-token definitions; probabilities are not automatically calibrated. API behavior may change across upstream revisions.

## 21. Why it matters scientifically

Does a reusable typed-classifier serving API make direct scoring practical beyond a research CLI?

## 22. Source provenance

Repository: [https://github.com/featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `none`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
