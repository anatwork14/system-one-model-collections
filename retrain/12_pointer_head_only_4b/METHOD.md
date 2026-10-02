# Pointer-head-only-4B

## 1. Status

Proposed ablation; not trained

## 2. Family

D — frozen-backbone pointer scoring

## 3. Base model

Model: `Qwen/Qwen3.5-4B-Base`  
Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`  
Parameter count: 4B class  
Base vs instruct: Base

## 4. Core idea

Reuse Kev candidate representation/readout with a fully frozen Qwen base and train only the pointer head.

## 5. Architecture

```text
state + candidates → frozen Qwen hidden states → trainable pointer head → candidate softmax
```

## 6. Input

Kev-format state/question/candidate rows, converted from controlled Nimble decisions.

## 7. Output

Candidate probability distribution.

## 8. Autoregressive?

No generated answer sequence.

## 9. Decision probability

Pointer-head score softmax across candidates.

## 10. Training

Pointer CE; head-only ablation.

## 11. Trainable modules

Pointer head only.

## 12. Frozen modules

All Qwen3.5-4B-Base backbone weights.

## 13. Training objective

Candidate cross-entropy over pointer scores.

## 14. Training data

Initial controlled set is the 2,676 Nimble train records, converted to Kev's row format; optional Kev decision-v7 extension is a separate experiment and must not be mixed into the first controlled result.

## 15. Training configuration

Same train IDs/split/seed as other controlled arms; proposed seed 17. Head optimizer settings are in configs/pointer_head.yaml and must be frozen before full run.

## 16. Published artifacts

Kev upstream pointer implementation as read-only source; local `pointer_head.pt` plus its recipe metadata will be written under `weights/head_only/`.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

After training, score the frozen Nimble holdout with `scripts/eval.sh`. The small local inference wrapper loads the pinned Kev pointer implementation and frozen base directly. No training executed.

## 19. Expected resources

Frozen BF16 4B base is about 9 GB of weights; plan for roughly 10–14 GB with single-row activations and head. Estimate only; training peak depends on hidden-state retention.

## 20. Known limitations

Head-only result depends on representation quality and exact Kev serialization/masks; comparison with Kev LoRA must share initialization, data, and evaluation.

## 21. Why it matters scientifically

Is a custom decision head sufficient, or is backbone adaptation needed?

## 22. Source provenance

Repository: [https://github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `controlled Nimble-derived decisions`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
