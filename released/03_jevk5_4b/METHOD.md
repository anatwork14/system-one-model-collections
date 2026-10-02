# JevK5-4B

## 1. Status

Published merged checkpoint

## 2. Family

C — distilled backbone with option-logit readout

## 3. Base model

Model: `Qwen/Qwen3.5-4B`  
Revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`  
Parameter count: 4B class  
Base vs instruct: post-trained model built from the explicit `Qwen3.5-4B-Base`; the Hub card does not name this checkpoint “Instruct”.

## 4. Core idea

Distilled LoRA is merged into Qwen3.5-4B; runtime uses a SemIf-style softmax over answer-letter logits with a calibration temperature.

## 5. Architecture

```text
state + typed question + candidates → merged Qwen → option-letter logits → temperature-scaled softmax
```

## 6. Input

Typed state/question/options, encoded using the project's prompt format.

## 7. Output

Probability for every candidate; no free-form answer required at runtime.

## 8. Autoregressive?

No generated answer sequence at runtime; single forward option-logit readout (multiple passes may be used for oversized option sets).

## 9. Decision probability

Softmax of allowed answer-letter next-token logits after published temperature calibration.

## 10. Training

LoRA decision distillation; teacher-generated decisions plus public replay. Current v0.3 reports 17,408 teacher questions and 30,052 public replay items.

## 11. Trainable modules

LoRA parameters during training; merged into released backbone.

## 12. Frozen modules

Base parameters except LoRA during training; entire merged checkpoint is fixed at inference.

## 13. Training objective

Option-letter cross-entropy/distillation as described by the tagged training recipe; v0.1/v0.2 settings are not asserted to describe current v0.3 exactly.

## 14. Training data

17,408 teacher questions (3,270 from Qwen3.6-27B and 14,138 from GPT-6 Luna) plus 30,052 human-labelled replay items from train splits of 26 public datasets. No dataset test/validation splits were used for v0.3 replay; teacher data is generated and subject to teacher-model terms. Public datasets retain their own licenses. The v0.3 training set is 47,460 rows. Distinguish generated local teacher files, upstream public source data, and the released checkpoint.

## 15. Training configuration

Version-specific. v0.1/v0.2 recipe reports rank 16, LR 3e-5, two epochs; consult exact pinned upstream tag/config before reproducing current v0.3.

## 16. Published artifacts

Merged HF checkpoint alibiserikbay/JevK5 and tokenizer/config; repo provides training/inference implementation.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

See scripts/smoke.sh and upstream README; call the package/server or direct scoring helper with shared base tokenizer assets as documented upstream.

## 19. Expected resources

The model card reports about 9 GB GPU memory in BF16; longer inputs and CUDA graph capture can add runtime memory. Inputs over 16,384 tokens are refused.

## 20. Known limitations

English-focused, option/context limits and temperature are version-specific; softmax calibration may shift with new domains and option counts; teacher labels may inherit model errors.

## 21. Why it matters scientifically

Does distillation plus calibrated option-logit readout improve typed decisions while retaining one-pass inference?

## 22. Source provenance

Repository: [https://github.com/allebee/jevk5](https://github.com/allebee/jevk5)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `alibiserikbay/JevK5`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `v0.3 teacher + public replay`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
