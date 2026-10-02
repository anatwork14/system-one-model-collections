# Clef-style Qwen3.5-4B

## Status

Local reproduction. No official Cloudflare Clef-4B checkpoint exists. Keep this method identified as `OURS_CLEF4B_*`; the official 9B release lives in `released/05_clef_flash_9b/`.

## Method

- Base: `Qwen/Qwen3.5-4B`, pinned workspace revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Inputs: shared state plus one or more runtime-defined `choice`, `score`, or `noul` questions.
- Output: one logit/probability distribution over each question's allowed options.
- Decision step: non-autoregressive; Qwen prefill feeds a joint schema routing head.
- Backbone: frozen original parameters.
- Adaptation: rank-256 language LoRA in the faithful-size arm; exact target modules were not disclosed by Cloudflare. The targets in this workspace are explicit local choices.
- New head: published Clef `JointSchemaHead` code with a fresh initialization at hidden size 2,560. No 9B trained head weights are resized or partially copied.
- Supervised objective: label-smoothed cross-entropy plus Brier loss. The Brier coefficient is a local hyperparameter, never represented as an official value.
- RL: deferred. Any later stage is an RLCD-like reproduction, not Cloudflare's exact unpublished RLCD.
- Data: open surrogate based on the workspace's Nimble train set, with evidence-certificate yes/no fields and train-only schema permutations. No Cloudflare data is included.

## Current readiness

The official code and head have been inspected. A fresh 4B-dimension head passes a synthetic mixed-question output-shape check. The initial data recipe creates 2,676 canonical records, group-splits source families into train/calibration/dev, reserves Nimble's 324-row evaluation set as test, and produces schema permutations from training records only.

The host currently exposes a GTX 1080 Ti (11 GB). The installed PyTorch CUDA 13 build is incompatible with its driver, and even a compatible build would not fit a 4B model plus rank-256 training state on this card. Full model training and probability benchmark results remain unrun; use a supported higher-memory CUDA host for those stages.

## Limitations

The initial dataset is synthetic/model-checked and not fully human reviewed. Evidence questions derive only from `supported`/`refuted` certificates; `unknown` facts are omitted because binary `noul` cannot represent unknown. Training data, base split, source labels, and objective coefficients differ from Cloudflare's undisclosed original recipe.
