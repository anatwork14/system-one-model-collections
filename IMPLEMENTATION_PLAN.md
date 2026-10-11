# Implementation Plan — Qwen3.5-4B → System-One Model Zoo

## Goal and boundaries

Create one reproducible experimental workspace comparing direct next-token scoring, SFT, candidate-only CE, contrastive representations, pointer heads, and full fine-tuning. Upstream repositories are treated as read-only source snapshots; model downloads live once under `shared/models/`, while method-specific adapters and heads remain in each method folder. No training is performed during workspace preparation.

## Repository map

```text
environment/                 dependency and host diagnostics
shared/models/               pinned Qwen3.5-4B and Qwen3.5-4B-Base
shared/datasets/              immutable source data and shared protocols
shared/benchmarks/            frozen benchmark definitions
released/00_semif/            frozen direct-logit baseline
released/01_simple_jev/       classifier API and scoring implementation
released/02_tev1_4b/          published Qwen3.5-4B SFT artifact
released/03_jevk5_4b/         published merged Qwen3.5-4B artifact
released/04_kev_4b/           published Qwen3.5-4B-Base LoRA + pointer head
released/05_clef_flash_9b/    official Cloudflare Clef-Flash 9B reference release
retrain/10_nimble_4b/         4B candidate-logit CE reproduction
retrain/11_clm_style_4b/      frozen-base contrastive-head reproduction
retrain/12_pointer_head_only_4b/ frozen-base pointer-head ablation
retrain/13_kev_fullft_4b/     full-backbone pointer upper-bound recipe
retrain/14_vlm_systemone_4b/  guarded V1 multimodal head-only experiment; later stages deferred
retrain/15_clef_style_4b/    Clef-style Qwen3.5-4B reproduction
retrain/16_adas1_multiexit_4b/ AdaS1 frozen layer-probe screen (P0)
evaluation/                   shared adapters, metrics, runner
results/                      released and controlled result records
```

Method folders include `METHOD.md`, `ARTIFACTS.lock`, and the applicable `upstream/`, `weights/`, `data/`, `configs/`, `scripts/`, and `results/` directories. `REPRODUCTION.md` and `DATASET.md` are added where relevant.

## Pinned model inputs

- `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- `Qwen/Qwen3.5-4B-Base`, revision `1001bb4d826a52d1f399e183466143f4da7b741b`.

The exact full base-model SHA for the second model expands the short `1001bb4d` pin in the initial notes. `shared/models/MODELS.lock` is the canonical record. Downloaded file hashes are added by `tools/fetch_all.sh` after successful downloads.

## Method matrix

| ID | Method | Base | Backbone updates | Decision head/readout | Objective |
|---|---|---|---|---|---|
| M0 | Raw Qwen | Qwen3.5-4B | none | generated letter | autoregressive generation |
| M1 | SemIf | Qwen3.5-4B | none | vocabulary option logits | none |
| M2 | Simple Jev | Qwen3.5-4B | none by default | vocabulary option logits | none |
| M3 | Tev1 | Qwen3.5-4B | LoRA | existing LM head | SFT/token CE |
| M4 | JevK5 | Qwen3.5-4B | distilled LoRA, merged | option-letter logits | option CE/distillation |
| M5 | Kev | Qwen3.5-4B-Base | LoRA | pointer head | pointer CE |
| M6 | Nimble reproduction | Qwen3.5-4B | LoRA | candidate-only LM logits | candidate CE |
| M7 | CLM-style reproduction | Qwen3.5-4B | frozen | state/action projections | InfoNCE |
| M8 | Pointer-only | Qwen3.5-4B-Base | frozen | pointer head | pointer CE |
| M9 | Kev-style full FT | Qwen3.5-4B-Base | full | pointer head | pointer CE |
| M10 | Clef-Flash reference | Qwen3.5-9B | published merged weights + head | joint schema routing | published SFT + RLCD (training recipe partly undisclosed) |
| M11 | Clef-style reproduction | Qwen3.5-4B | frozen base + local rank-256 LoRA | freshly initialized published joint head | smoothed CE + Brier; optional RLCD-like later |

The training launcher uses a separate M10–M16 ID namespace, so launcher M16 is not the M16 row of any future model-zoo matrix. Its current implementation is only the AdaS1 P0 layer-probe screen. The pasted proposal's original plan to select exits on Nimble's frozen 324-row outer set was rejected to preserve that set: P0 trains and screens on the existing family-disjoint 2,268/408 inner train/dev split. See [M16 method card](retrain/16_adas1_multiexit_4b/METHOD.md) and [research-gap review](retrain/16_adas1_multiexit_4b/RESEARCH_GAP.md).

## Experiment sequence

1. Fetch/pin sources and weights; validate artifacts and shared symlinks.
2. Run the same released smoke decision through all five released methods and save raw outputs plus timing/memory fields. This is infrastructure validation, not a benchmark.
3. Keep Nimble's 2,676 source training rows and 324-row outer evaluation set immutable; derive one family-disjoint inner train/dev split from the source training rows for all controlled arms.
4. Re-run pointer-only, LoRA+pointer, and CLM-style arms on the same inner train rows, dev families, seed, and evaluation protocol. Existing runs are exploratory because they repeatedly used the outer evaluation set.
5. Run full fine-tuning only after smaller arms have validated configs and reload paths.
6. Scale the strongest methods to the approximately 38K Tev1 `new-v1` mixture, with the exact same records per arm.
7. Add open-world known/novel/unknown splits only after the controlled text comparison.
8. Port the winning decision mechanism to the multimodal model in a separate later phase.
9. Keep the official Cloudflare Clef-Flash 9B artifact distinct from the local Clef-style 4B reproduction; do not label the latter as an official checkpoint.
10. Run M16 P0 frozen layer probes on the 2,268/408 train/dev split only. Proceed to teacher and multi-exit training only if inner-dev results support useful intermediate exits; keep outer test untouched until the architecture and selection rules are fixed.

## Data rules

Upstream data is never edited in place or used as an implicit writable workspace. Keep source snapshots under `upstream/`, immutable copies under `data/raw/`, derived records under `data/processed/`, split IDs under `data/splits/`, and provenance/hash/license records under `data/manifests/`. The initial controlled dataset is Nimble's published 2,676 train and 324 frozen evaluation examples. Nimble labels are synthetic/model-checked and not human reviewed. Never tune on its frozen holdout.

## Probability and token checks

For every candidate-logit method, record tokenizer IDs and verify intended option tokens. Validate output option membership, finite scores, and probability sum tolerance. Softmax normalization is not calibration; report accuracy alongside NLL, Brier score, ECE, and AURC where labels support them.

## Completion state

Workspace structure, method cards, configs, scripts, result schemas, and provenance files are prepared. Historical exploratory training artifacts exist for Nimble, CLM-style, and pointer-head-only arms; they do not count as clean controlled results because the outer evaluation split was consulted. Corrected family-disjoint inner splits and launchers are prepared, but those clean runs and full fine-tuning remain unrun. Smoke inference and standardized benchmark measurements are still pending. Clef-style 4B training remains unrun. M14 V1 now has a guarded trainer, prediction adapter, leakage audit, and progress/resume artifacts; it remains blocked on licensed image-decision data and a compatible CUDA runtime. M16 P0 code is prepared but has not run; all later AdaS1 stages remain conditional on its result.
