# Kev-style Full-FT-4B

## 1. Status

Proposed upper-bound reproduction; not trained

## 2. Family

D — pointer scoring with full backbone fine-tuning

## 3. Base model

Model: `Qwen/Qwen3.5-4B-Base`  
Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`  
Parameter count: 4B class  
Base vs instruct: Base

## 4. Core idea

Keep Kev pointer architecture and train every Qwen backbone parameter plus the pointer head.

## 5. Architecture

```text
state + candidates → fully trainable Qwen → pointer head → candidate softmax
```

## 6. Input

Kev-format state/question/candidate rows.

## 7. Output

Candidate probability distribution.

## 8. Autoregressive?

No answer-text generation.

## 9. Decision probability

Pointer-head score softmax.

## 10. Training

Full fine-tuning with pointer candidate CE.

## 11. Trainable modules

All backbone weights and pointer head.

## 12. Frozen modules

None, aside from any explicitly frozen vision modules in the exact text path; default recipe trains the whole text backbone.

## 13. Training objective

Candidate pointer cross-entropy.

## 14. Training data

Same 2,676 Nimble train rows and frozen 324 evaluation rows for the first controlled comparison; labels synthetic/model-checked.

## 15. Training configuration

Controlled recipe: seed 17, one epoch, LR 1e-5, head LR 1e-4, batch 1, accumulation 8, BF16, and gradient checkpointing. The pinned Kev trainer keeps FP32 optimizer masters in host memory on one device; use its FSDP2 path only when deliberately launching multiple ranks. Do not launch before smaller arms pass their data/model wiring checks.

## 16. Published artifacts

Kev upstream pointer implementation; local full checkpoint under weights/fullft.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

After training, score the frozen Nimble holdout with `scripts/eval.sh`; for a service, start `python -m kev.serve --run weights/full_ft`. No training executed.

## 19. Expected resources

Full BF16 weights alone are about 9 GB; gradients and optimizer state raise total memory well beyond a single 24 GB GPU unless state is sharded/offloaded. Plan for multi-GPU FSDP or CPU offload; no local peak measurement is available.

## 20. Known limitations

Expensive and may overfit the small synthetic set; full fine-tuning comparison is only meaningful with identical initialization/data/splits and independent checkpoint evaluation.

## 21. Why it matters scientifically

What quality ceiling does full backbone adaptation add over LoRA or head-only pointer training?

## 22. Source provenance

Repository: [https://github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `controlled Nimble-derived decisions`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
