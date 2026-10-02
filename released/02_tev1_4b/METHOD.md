# Tev1-4B

## 1. Status

Published checkpoint

## 2. Family

B — autoregressive answer-token SFT

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class; post-trained multimodal checkpoint  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

LoRA supervised fine-tuning specializes the model to choose a letter for a structured decision. The standard LM head is retained.

## 5. Architecture

```text
state + question + labeled options → Qwen backbone + LoRA → next answer letter → application maps letter to option
```

## 6. Input

System instruction and structured state, question, and 2–24 labeled options.

## 7. Output

One option letter (then application maps it to the semantic key).

## 8. Autoregressive?

Yes, autoregressive; intended output is one letter, not a probability vector.

## 9. Decision probability

If available, token log-probabilities can be exposed for letter choices; ordinary generated answer itself is the published interface. Do not assume calibrated option probabilities.

## 10. Training

LoRA SFT, one epoch in published starting recipe.

## 11. Trainable modules

LoRA adapters; existing LM head is used and not described as a new head.

## 12. Frozen modules

Base weights outside LoRA.

## 13. Training objective

Full-vocabulary next-token CE over supervised answer output.

## 14. Training data

Published new-v1: 37,840 train and 4,568 development examples. Sources include MultiNLI, BoolQ, Banking77, AG News, SST-5, generated policy/routing tasks, and synthetic research classification. Labels are a mix of source labels and construction labels, not uniformly human-reviewed. See `data/DATASET.md` and upstream `DATA_SOURCES.md` for license caveats; some source license metadata is unknown or unspecified.

## 15. Training configuration

Published preview recipe: rank 8, alpha 16, LR 5e-5, batch 8, gradient accumulation 1, sequence length 2048, 1 epoch, packing, seed 42; target training environment is Together H100. Recipe may not match a historical run byte-for-byte.

## 16. Published artifacts

Full model checkpoint from togethercomputer/Tev1-4B-experimental; tokenizer/config included in HF repo; base source repo and recipe under upstream.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

See scripts/smoke.sh; use Transformers generation or the upstream example client. Local generation command is prepared in scripts/infer.py.

## 19. Expected resources

Published snapshot is about 9.34 GB on Hub. Plan for roughly 10–14 GB GPU memory in BF16 for a short single-request run; this is an estimate and depends on serving engine/context.

## 20. Known limitations

Autoregressive output adds decoding; one-letter output has no inherent typed calibration; benchmark is not directly comparable to non-generative probability readout without protocol alignment.

## 21. Why it matters scientifically

Does decision-focused LoRA SFT improve answer-letter selection over frozen direct logits?

## 22. Source provenance

Repository: [https://github.com/togethercomputer/tev1](https://github.com/togethercomputer/tev1)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `togethercomputer/Tev1-4B-experimental`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `new-v1`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
