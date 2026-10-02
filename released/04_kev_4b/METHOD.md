# Kev-4B

## 1. Status

Published adapter and pointer head

## 2. Family

D — candidate pointer scoring

## 3. Base model

Model: `Qwen/Qwen3.5-4B-Base`  
Revision: `1001bb4d826a52d1f399e183466143f4da7b741b`  
Parameter count: 4B class; model card reports 4.7B total including vision tower  
Base vs instruct: Base

## 4. Core idea

Qwen hidden states for decision marker and candidates feed a custom pointer head. Final decisions come from candidate scores, not LM vocabulary logits.

## 5. Architecture

```text
shared state + question/decision marker + candidate spans → Qwen + LoRA hidden states → pointer head → candidate softmax
```

## 6. Input

State with one or more typed questions and candidate descriptions serialized using Kev's delimiter format.

## 7. Output

Probability distribution over candidates/typed answers.

## 8. Autoregressive?

No answer text generation; one forward decision pass.

## 9. Decision probability

Softmax of pointer-head candidate scores.

## 10. Training

LoRA plus pointer-head supervised training; published recipe uses decision-v7 and two epochs; later release also includes a delta update.

## 11. Trainable modules

LoRA rank 16 (reported 33.8M parameters) and pointer head.

## 12. Frozen modules

Base parameters outside LoRA during published PEFT training.

## 13. Training objective

Pointer/candidate cross-entropy.

## 14. Training data

decision-v7 (10,000 examples across ten public datasets, plus 896 generated policies and 1,680 generated rule examples) plus the current skills delta (11,320 new records and 4,000 replay). The delta combines 6,000 programmatically labelled hard-v1 examples and 5,320 devtools-v1 examples from CodeReviewer, CommitPackFT, FlakeFlagger, and Aegis. Source label quality varies; generated labels are not human ground truth. Per-source licenses and provenance are in the upstream model card/manifests.

## 15. Training configuration

Recipe: LoRA r=16 over attention, MLP, and DeltaNet projections; LR 5e-5, two epochs; later delta is separately documented upstream. Exact checkpoint revision is in the artifact lock.

## 16. Published artifacts

HF adapter and pointer head, plus Qwen3.5-4B-Base shared base. Model card files and tokenizer/config metadata must be captured by fetch manifest.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

See scripts/smoke.sh and upstream model card/server entry point; adapter and head load on weights/base_model.

## 19. Expected resources

Published model card reports about 9 GB GPU memory for BF16 weights and about 14 GB with server batching buffers. Adapter loading may briefly hold a second base copy; measure the selected load path.

## 20. Known limitations

Requires custom architecture/runtime and exact row/mask serialization; candidate descriptions and delimiter/token IDs matter; generated data and calibration can limit OOD behavior.

## 21. Why it matters scientifically

Does a learned pointer readout with modest backbone adaptation outperform vocabulary-token scoring?

## 22. Source provenance

Repository: [https://github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)  
Requested ref: `main`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `jaredpalmer/kev-4b`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `decision-v7 + later delta`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
