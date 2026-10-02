# VLM System-One 4B

## 1. Status

NOT_STARTED — placeholder only

## 2. Family

F — multimodal decision model

## 3. Base model

Model: `Qwen/Qwen3.5-4B multimodal checkpoint (exact selected revision to be pinned later)`  
Revision: `deferred`  
Parameter count: 4B class plus bundled vision components  
Base vs instruct: deferred; select and pin a multimodal checkpoint before this experiment starts.

## 4. Core idea

Port the best validated text decision mechanism to image-conditioned decisions after text experiments.

## 5. Architecture

```text
image + text state + typed candidates → multimodal Qwen → selected decision readout
```

## 6. Input

Image(s), textual state/question, and typed candidate choices.

## 7. Output

Typed decision probabilities.

## 8. Autoregressive?

To be selected with winning text mechanism.

## 9. Decision probability

To be selected and explicitly calibrated/evaluated.

## 10. Training

Not started. Planned V0 frozen VLM/direct logits; V1 frozen VLM/head; V2 language LoRA/head; V3 connector + language LoRA/head; V4 upper vision blocks; V5 full VLM fine-tuning.

## 11. Trainable modules

None selected yet.

## 12. Frozen modules

None selected yet.

## 13. Training objective

To be selected after text phase.

## 14. Training data

No dataset selected. Keep image train/validation/test splits and image provenance/license explicit.

## 15. Training configuration

Not started.

## 16. Published artifacts

None.

## 17. Local artifacts

- `upstream/`: pinned source repository, treated as read-only.
- `weights/`: method checkpoint/adapter/head, or symlink to `shared/models/`.
- `data/`: raw/processed/splits/manifests; source data is not edited in place.
- `configs/`, `scripts/`, `results/`: local reproducibility assets and outputs.
- Artifact download status and exact revisions: `ARTIFACTS.lock`.

## 18. Inference

Not available.

## 19. Expected resources

Not estimated.

## 20. Known limitations

Deferred; image-text leakage, visual preprocessing, and multimodal calibration need separate protocol.

## 21. Why it matters scientifically

Can the best text decision interface extend to image-grounded System-One decisions?

## 22. Source provenance

Repository: [none](none)  
Requested ref: `none`  
Resolved source commit: recorded by `tools/fetch_all.py` after network fetch; see `ARTIFACTS.lock`.  
Checkpoint: `none`  
Checkpoint Hub revision: resolved to a full immutable commit SHA during fetch where applicable.  
Data recipe: `none`  
Downloaded at: recorded after successful fetch.  
Local changes: workspace metadata, wrappers, and configs only; upstream files are not modified.
