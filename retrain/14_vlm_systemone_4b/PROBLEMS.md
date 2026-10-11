# M14 problems and readiness gaps

**Audited:** 2026-10-06  
**Current state:** V1 training code exists, but a valid M14 training run cannot start in this workspace. No data has been selected or trained on, and no M14 quality result is available.

This report separates hard launch blockers from implementation, evaluation, and research gaps. “Implemented” means code is present; it does not mean the behavior has been verified on a compatible GPU with real M14 data.

## Hard blockers

| Priority | Problem | Evidence and consequence | Needed to clear it |
|---|---|---|---|
| Blocker | No M14 dataset exists | `data/manifests/manifest.json` has `status: not_ready`, an empty `sources` list, and null train, model-validation, calibration, test, and OOD splits. There are no normalized M14 records or image files. The trainer correctly refuses to run. | Select an image-decision source; document image and dataset terms, provenance, labels, transformations, and hashes; prepare all five splits; pass the full audit. |
| Blocker | Image-use rights are unverified | Existing Flowers-102 records carry `VERIFY_UPSTREAM_TERMS`. The validator can require explicit attestations but cannot review or establish legal rights. | Review the actual source and image terms, record constraints and attribution, and set attestations only after that review. Do not infer training rights from a dataset code license. |
| Blocker | Available data does not match the experiment | Nimble is text-only. Flowers-102 is a narrow classification benchmark rather than broad, image-grounded decision data, and its terms remain unresolved. | Use a suitable licensed decision dataset, or explicitly scope a small pilot as a narrow classification-to-candidate experiment and keep its claims narrow. |
| Blocker on this host | CUDA cannot initialize | The workspace uses PyTorch `2.14.1+cu130`, while the installed NVIDIA driver reports CUDA 12.2. PyTorch reports no usable CUDA device. | Use a PyTorch/CUDA build compatible with the installed driver or update the driver in the training environment. |
| Blocker on this host | GPU does not meet the launch gate | The detected GTX 1080 Ti has 11 GiB; V1 enforces a 23 GiB runtime-visible minimum. | Run the V1 pilot on a compatible GPU with at least 23 GiB visible to PyTorch, then record actual peak memory. This is a planning gate, not an empirically validated guarantee. |
| Verification gap | Full model forward is untested | The pinned processor successfully creates image-conditioned tensors, but the Qwen3.5 multimodal backbone has not completed a real forward or optimizer step in this environment. | On the target GPU, run a one-question forward/backward smoke test before the full dataset run; verify finite hidden states, loss, gradients, checkpoint reload, and prediction. |

The exact host findings and current manifest are recorded in [STATUS.md](STATUS.md) and [the data manifest](data/manifests/manifest.json).

## Data quality and leakage limits

The current validator is a useful fail-closed gate, but it cannot make a poorly constructed dataset safe by itself.

- Rights, provenance, attribution, and the assertion that test data was excluded from human model selection are manifest attestations. Software cannot determine whether those attestations are truthful or whether a source's terms permit the intended use.
- Group and family separation depends on the supplied `group_id`, `source_family`, and source-record IDs being complete and correct. The validator cannot infer that two differently named sources share an underlying image, workflow, or originating example.
- Image near-duplicate screening uses a perceptual hash threshold. It can miss transformed or semantically identical images, and similar but distinct images may need manual review. Text screening detects exact normalized long prompts only; it is not semantic deduplication.
- The schema checks required fields and candidate/target membership, but does not validate every domain-specific rule—for example, whether score candidates are correctly ordered, labels are human-verified, or `type` values follow a closed vocabulary.
- No source or split exists today, so the audit has only been exercised with temporary synthetic fixtures. It has not been run against real M14 data.

Before training, inspect the audit report and manually review suspicious duplicate pairs and source-family assignments. A passing programmatic audit is necessary, not proof that labels, rights, or split semantics are correct.

## V1 implementation and reproducibility gaps

- The V1 trainer is the only implemented M14 training stage. It freezes the full multimodal backbone and trains a fresh linear candidate head. V0 baselines, V2 language LoRA, V3 merger adaptation, V4 vision-block adaptation, and V5 full fine-tuning remain proposal/configuration entries, not executable training paths.
- The V1 recipe is now aligned on five maximum epochs, no scheduler, and patience-based early stopping. The trainer accepts one Brier weight per invocation; the listed ablation values require separate runs and are not an automated sweep.
- The config lists Brier-weight ablation values, while the trainer runs one selected weight per launch. The CLI permits changing the weight, but there is no managed sweep or comparison report.
- The V1 loop updates once per question and scores each candidate independently. It reprocesses the same image/state for every candidate and does not batch questions or cache shared image features. This limits throughput and makes real run time depend heavily on image size and candidate count.
- The trainer rejects branches over 2,048 tokens and caps image processing at 786,432 pixels. These are local launch settings, not validated limits for the eventual data distribution; long states or high-resolution OCR/screens may be rejected or lose useful detail.
- The input serialization now includes the record's `type` field; the trainer records prompt version `m14-v1-typed-candidate-v2` in `run_config.json`.
- A seed is recorded, but there is no completed multi-seed run or checked-in end-to-end test proving repeatability on the target CUDA stack. Kernel and runtime differences can still change exact results.

See [the trainer](scripts/train.py), [the variant config](configs/variants.yaml), and [the proposed stages](METHOD.md) when resolving these gaps.

## Evaluation and scientific gaps

- Checkpoint selection uses model-validation NLL plus `0.25 × Brier`, which is appropriately separate from test. However, post-hoc temperature fitting is not implemented, even though the method design calls for the calibration split.
- The current prediction path scores one explicitly selected split under the full image+text condition. Text-only, shuffled-image, image-only, corruption, candidate-order, and modality-shift evaluations are not implemented as a complete evaluation suite.
- The shared metric path can calculate common classification/calibration/selective metrics from prediction probabilities, but M14 prediction records do not preserve source, task type, or domain metadata needed for reliable per-source/per-task slices. The runner also does not produce the proposed paired confidence intervals or multi-seed comparisons.
- OOD has no concrete source or frozen definition yet. Until OOD families and evaluation conditions are chosen before training, “visual grounding,” “robustness,” and “generalization” claims cannot be supported.
- No train/dev/test results, latency baseline, peak-memory measurement, or accuracy/calibration comparison exist. The current processor smoke test is an infrastructure check only.
- The research contribution is not established by implementing a multimodal head alone. The proposal notes overlap with existing multimodal System-One methods; novelty requires a literature comparison and a result-backed question such as transfer, adaptation depth, visual reliance, or calibration under modality shift.

## Recommended order to resolve the problems

1. Choose and review the data source and intended claims. If starting from Flowers-102, first resolve its terms and label it clearly as a narrow pilot rather than a general decision dataset.
2. Build normalized records and freeze family/group-disjoint train, model-validation, calibration, test, and OOD IDs. Verify target labels and source metadata manually on a sample.
3. Reconcile the executable V1 hyperparameters with `configs/variants.yaml`; decide whether to implement warmup/scheduling or remove those claims from the config.
4. Install a compatible CUDA/PyTorch stack on a GPU meeting the memory gate. Run a one-question forward/backward and checkpoint-reload test; profile several image sizes and candidate counts.
5. Run the full leakage audit, review its output, and launch V1 with step-level progress and checkpointing.
6. Implement calibration and the modality/order/OOD evaluation conditions before using the test split. Freeze the protocol first, then report repeated-seed results and uncertainty.

Do not mark M14 complete, publish a checkpoint as validated, or claim visual grounding until the blockers are cleared and the corresponding evaluation is run.
