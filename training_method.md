# Training methods in this workspace

This document describes the repository's locally maintained training experiments, their research motivation, data, and optimization recipes. It also identifies the published model recipes kept in `released/` as comparison references. The runnable registry and commands live in [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md), and each method's source-of-truth card is linked below.

“Runnable” means a training and evaluation path is registered and implemented. It does not mean a clean run has completed. At the time of this inventory, the method cards do not claim completed clean benchmark runs. The runbook uses an explicit CUDA preflight before training.

The workspace uses two ID schemes. The implementation-plan matrix uses M0–M11 for the full model-zoo comparison; the training launcher uses M10–M16 for local training experiments. The IDs overlap but refer to different things. In particular, a public checkpoint can be loaded and evaluated without training it in this workspace. The model's published training history is described where known, while local inference over released weights is not counted as a local training run.

## Full model-zoo comparison: implementation-plan IDs

This is the broader comparison, including methods that only run inference from frozen or published public weights. “Local training” below means the workspace has a training recipe or reproduction; the existence of a public trained checkpoint does not imply that this workspace trained it.

| Plan ID | Method | Comparison role in this workspace | Data and training context |
|---|---|---|---|
| M0 | Raw Qwen | Planned inference baseline: frozen public Qwen3.5-4B generates an option letter | No local training or decision-specific data. The implementation plan includes it, but the workspace has no separate M0 method package in the evaluation registry yet. |
| M1 | SemIf / OpenJev | Frozen Qwen direct option-token scoring | No training or training data; evaluates allowed vocabulary logits. |
| M2 | Simple Jev | Frozen Qwen classifier/scorer by default | No local training for baseline serving. Upstream offers a separate optional RFDT workflow. |
| M3 | Tev1 | Run the published Qwen3.5-4B SFT checkpoint | Already trained public weights. The published new-v1 recipe uses 37,840 train and 4,568 dev examples; rank 8 LoRA, one epoch, answer-token CE. |
| M4 | JevK5 | Run the published merged Qwen3.5-4B checkpoint | Already trained public weights. v0.3 combines 17,408 teacher questions and 30,052 replay records; local workspace does not retrain it. |
| M5 | Kev | Run the published Qwen3.5-4B-Base LoRA plus pointer head | Already trained public adapter/head. Published recipe uses decision-v7 data and a later skills delta/replay; local workspace does not retrain the released artifact. |
| M6 | Nimble reproduction | Train local candidate-only logit model | Local controlled Nimble split; see training-launcher M10 below. |
| M7 | CLM-style reproduction | Train local contrastive state/action projections | Local controlled Nimble-derived pairs; see launcher M11. |
| M8 | Pointer-only | Train local pointer head over a frozen base | Local controlled Nimble-derived Kev rows; see launcher M12. |
| M9 | Kev-style full FT | Train local full-backbone pointer model | Local controlled Nimble-derived Kev rows; see launcher M13. |
| M10 | Clef-Flash 9B reference | Run official public Qwen3.5-9B merged weights and joint head | Official training used internal synthetic data, smoothed CE + Brier, then RLCD; exact data and recipe are partly undisclosed. This is inference over a published artifact locally. |
| M11 | Clef-style 4B reproduction | Train a fresh local 4B head; optional local LoRA | Local Nimble-derived surrogate; see training-launcher M15. It is not an official Cloudflare checkpoint. |

For the public-weight rows M0–M5 and M10, the local question is “how does this available model behave under the same evaluation?”, not “can we reproduce its training?”. Their published recipe and data details are context for interpreting the checkpoint. M1 and M2 are zero-training decision methods; M3–M5 and M10 are published trained models used as ready-made weights. M0 is the unadapted generative baseline. Smoke inference checks that these artifacts load and return valid outputs; smoke results are infrastructure checks, not a benchmark result.

## Shared data and evaluation protocol

The controlled local arms M10–M13 start from the Nimble original-2676 dataset: 2,676 source training decisions across ten domains and Choice, Boolean/Noul, and Score tasks, plus a separate frozen 324-decision evaluation set. Labels are described by the source as synthetic and model-checked; they have not all been reviewed by people. Treat conclusions as controlled-method comparisons on a small surrogate, not as evidence of production task quality. Dataset provenance and hashes are in [the Nimble dataset card](shared/datasets/nimble/DATASET.md) and its manifest.

For M10–M13, the 2,676 source training records are deterministically partitioned by whole source family into 2,268 training rows and 408 inner-dev rows (28 and 6 families, respectively). Each method receives the same records in its own serialization. The inner-dev split supports checkpoint selection or diagnostics; source-family and source-ID checks prevent overlap. The 324-record outer evaluation split is excluded from training and selection. M15 makes its own family-group split of the same 2,676 canonical training records: 2,150 train, 250 calibration, and 276 dev; it preserves all 324 outer rows as test. Its split and label construction are described in [its dataset card](retrain/15_clef_style_4b/data/DATASET.md).

Scores from different methods are not a clean leaderboard by default. Input serialization, probability readout, and in M15 the expanded multi-question targets differ. Compare methods on matched records, target definitions, and evaluation procedures. The standard run report computes common classification, calibration, and selective-decision metrics where probability distributions are available. It does not infer routing, open-set, continual-learning, end-to-end system, cost, or robustness results without their required data.

## Local training experiments

| ID | Method | Adaptation | Registry state |
|---|---|---|---|
| M10 | Nimble candidate-logit CE | Qwen LoRA, rank 16 | Runnable |
| M11 | CLM-style contrastive candidate scoring | Frozen Qwen; state/action projections | Runnable |
| M12 | Kev-style pointer head-only | Frozen Qwen; train pointer head | Runnable |
| M13 | Kev-style full fine-tuning | Train Qwen and pointer head | Runnable, resource-intensive |
| M14 | Visual System-One candidate scoring | V1 fresh head over frozen VLM | V1 implemented; launch gated by licensed data and CUDA |
| M15 | Clef-style joint schema head | Head-only default; optional language LoRA | Supervised stage runnable |
| M15-RLCD | RLCD-like follow-on | Not implemented | Deferred |
| M16 | AdaS1 layer-probe screen | Frozen Qwen; identical lightweight probes at candidate exits | P0 probe runnable; all adaptive training/serving stages deferred |

### M10 — Nimble candidate-only language-model logits

**Method.** M10 adapts the public Nimble Qwen3.5-9B candidate-logit recipe to the pinned Qwen3.5-4B checkpoint (`851bf6e…`). It serializes state/context, a decision question, and the allowed choices, then reads the LM logits for the allowed candidate tokens and applies a softmax over only those candidates. It does not generate an answer sequence. The objective is candidate-only cross-entropy: the gold candidate competes with other allowed candidates, not every token in the vocabulary.

**Motivation.** This is the direct test of whether decision-specific LoRA training with a constrained candidate loss improves typed decisions over frozen option logits or ordinary answer-token SFT. It also gives a light adaptation baseline against the custom heads in M12–M13.

**Data.** Uses the shared Nimble-derived 2,268/408 family-disjoint train/dev split. The outer 324 records are never supplied to the trainer. The upstream labels are synthetic/model-checked. Token exports must match the pinned 4B tokenizer; the launcher prepares the inner split before training.

**Training recipe.** LoRA rank 16, learning rate `5e-5`, seed 17, micro-batch 2, gradient accumulation 4 (effective batch 8), maximum length 2,048, BF16, and one epoch. The step count is derived from the actual inner training row count, with warmup at about 10% of the optimizer steps. Base weights outside LoRA remain frozen. See [method card](retrain/10_nimble_4b/METHOD.md), [config](retrain/10_nimble_4b/configs/train_4b.yaml), and [trainer script](retrain/10_nimble_4b/scripts/train.sh).

**Status and caveat.** The clean corrected run has not been claimed complete. Older three-epoch VKong artifacts used the released outer evaluation set and are exploratory, not valid held-out results; do not resume those checkpoints.

### M11 — CLM-style state/action contrastive scoring

**Method.** The entire Qwen3.5-4B backbone is frozen. The trainer encodes a state prompt and each candidate action prompt, takes last-token hidden vectors, and learns separate 512-dimensional state and action projections plus a learned temperature. It normalizes the projections and scores candidates by scaled dot product, then returns a candidate softmax. Frozen backbone vectors are cached before fitting the small projection heads.

**Motivation.** This asks whether a frozen general-purpose representation can align a state with correct candidate descriptions, including newly supplied candidate text, without modifying billions of backbone parameters. It tests representation alignment and candidate flexibility against direct token scoring (M10) and pointer scoring (M12).

**Data.** Positive state-to-correct-action pairs are derived from the same 2,268 inner-training decisions. The other candidates in a decision act as negatives. The 408 family-disjoint inner-dev decisions select the checkpoint by mean loss, with patience 5. The released 324-record test set is not loaded. Original labels remain synthetic/model-checked.

**Training recipe.** Up to 20 epochs; AdamW learning rate `1e-4`; projection dimension 512; seed 17; one decision per update; maximum input length 2,048. The base is Qwen3.5-4B at the pinned workspace revision, fully frozen. The method is a small controlled adaptation, not a reproduction of CLM's original large-scale training. See [method card](retrain/11_clm_style_4b/METHOD.md), [config](retrain/11_clm_style_4b/configs/clm.yaml), and [trainer](retrain/11_clm_style_4b/scripts/train_contrastive.py).

**Status and caveat.** A clean inner-dev rerun has not been claimed. Quality depends strongly on candidate construction and negative examples; this controlled pair set is small.

### M12 — Kev-style pointer head-only

**Method.** M12 uses Kev's candidate serialization and pointer readout with a frozen Qwen3.5-4B-Base backbone (`1001bb4…`). A pointer head consumes the model's hidden representations for the state/decision and candidate spans, produces one score per candidate, and normalizes scores with a candidate softmax. Only the pointer head is trained; all Qwen parameters are frozen. The implementation caches per-record features before head optimization.

**Motivation.** This isolates the contribution of a structured candidate pointer head from backbone adaptation. Compared with M10, it tests a learned readout over candidate representations rather than candidate-token logits. Compared with M13, it measures whether adapting the backbone is necessary.

**Data.** The same controlled 2,268/408 Nimble inner train/dev split is converted into Kev-format rows. The 324 outer evaluation decisions are excluded. An optional Kev decision-v7 data extension is a separate experiment and must not be blended into the controlled baseline.

**Training recipe.** Pointer candidate cross-entropy; head dimension 256; seed 17; learning rate `1e-4`; weight decay `0.01`; up to 20 epochs; micro-batch 1; state context limit 2,048 with the Kev branch/packed limits. Checkpoint selection uses inner-dev mean record loss with patience 5. See [method card](retrain/12_pointer_head_only_4b/METHOD.md), [config](retrain/12_pointer_head_only_4b/configs/pointer_head.yaml), and [trainer](retrain/12_pointer_head_only_4b/scripts/train_head_only.py).

**Status and caveat.** The corrected training path exists, but a clean run is not claimed. Historical VKong output selected epochs on the outer 324-row evaluation set and is invalid as a held-out estimate.

### M13 — Kev-style full fine-tuning

**Method.** M13 keeps Kev's text serialization, pointer head, and candidate cross-entropy, but updates every Qwen3.5-4B-Base backbone parameter as well as the pointer head. It is the adaptation-depth upper bound for the M12 pointer-head ablation and complements M10's LoRA arm.

**Motivation.** It measures the potential quality gain from allowing the complete backbone to specialize to the decision task, at the cost of substantially higher compute and overfitting risk. This is only an informative upper-bound comparison when initialization, data, serialization, and evaluation match the lighter arms.

**Data.** Uses the identical 2,268 Kev-format inner-training rows and 408 family-disjoint dev rows as M12. Data hashes and split manifests are verified before launch. The fixed one-epoch trainer does not select the checkpoint on dev; dev is diagnostic. The 324 outer rows are not loaded by training.

**Training recipe.** One epoch, seed 17, BF16, micro-batch 1, gradient accumulation 8, backbone learning rate `1e-5`, pointer-head learning rate `1e-4`, gradient checkpointing, and length sorting. The single-device path keeps FP32 optimizer master state in host memory; multi-rank FSDP2 is available for deliberate distributed runs. See [method card](retrain/13_kev_fullft_4b/METHOD.md), [config](retrain/13_kev_fullft_4b/configs/full_ft.yaml), and [trainer launcher](retrain/13_kev_fullft_4b/scripts/train.sh).

**Status and caveat.** This is a prepared recipe, not a completed training result. The 4B model's BF16 weights alone are around 9 GB; gradients and optimizer state require much more memory. Plan for sharding or offload and record the actual resource use. Small synthetic data makes overfitting likely.

### M14 — multimodal Visual System-One candidate scoring

**Method.** M14 is a multimodal extension. The implemented V1 scores each candidate independently from images, state, question, and candidate text with a fresh linear scalar head over Qwen3.5 multimodal hidden states. The backbone, vision tower, and merger are frozen. Candidate scores become per-question probabilities via softmax; temperature fitting is a separate proposed calibration step. Later stages propose language LoRA, then optionally vision merger/top vision blocks and a full VLM upper bound.

**Motivation.** Its main questions are whether text decision training transfers to image-grounded decisions, what adaptation depth is needed, and whether predictions truly rely on visual evidence. It proposes text-only, correct-image, shuffled-image, and image-only tests to separate visual grounding from accepting multimodal inputs.

**Data.** No licensed image-decision dataset is selected. Flowers-102 in the workspace is not cleared as training data and is too narrow for the proposed mixture. Before training, M14 needs an auditable dataset with image rights, source/group IDs, stable candidates and labels, deduplication, family/image-group-disjoint train/dev/calibration/test partitions, and an OOD evaluation set. Do not treat suggested future dataset sizes or mixture percentages as requirements.

**Training plan and status.** V1 is implemented with CE + Brier loss, model-validation selection by `NLL + 0.25 × Brier`, step-level JSON progress, ETA/VRAM fields, and resumable `last.pt` plus selected `best.pt`. It opens train and model-validation labels only; calibration, test, and OOD are separate. V2 adds language LoRA; V3/V4 progressively unfreeze vision components only when evidence identifies a visual bottleneck; V5 is full fine-tuning. Those later stages are design-only. M14 is registered in the launcher, but launch is blocked because the image dataset manifest is not ready and the current host has an incompatible CUDA driver/runtime. Pillow/torchvision are installed and the processor image path has passed a tensor smoke check. See [reconciled method card](retrain/14_vlm_systemone_4b/METHOD.md) and [proposal](retrain/14_vlm_systemone_4b/METHODOLOGY_PROPOSAL.md).

### M15 — Clef-style joint schema head

**Method.** M15 adapts the published Clef-Flash joint schema-head architecture to Qwen3.5-4B. One record can contain multiple runtime-defined Choice, Score, or Boolean/Noul questions. A fresh head emits logits over each question's valid options without generating answer text. The 4B head is freshly initialized at hidden size 2,560; no weights from the official 9B Clef head are resized or copied. The supervised loss combines label-smoothed cross-entropy and Brier loss. The default runnable stage is head-only with the backbone frozen; an optional stage adds rank-256 language LoRA while keeping vision frozen.

**Motivation.** M15 tests whether a schema-aware head can answer several typed fields from shared state and improve probability quality through joint routing and a calibration-sensitive loss. It also tests a controlled 4B local reproduction of architectural ideas from Clef-Flash. It is not an official Cloudflare Clef-4B checkpoint or an exact reproduction: the original training records and several recipe details are unpublished.

**Data.** The local surrogate derives 2,676 canonical train records from Nimble and adds binary evidence questions only for certificate atoms marked supported or refuted; unknown facts are omitted because the binary target cannot represent unknown. Whole source families are assigned to 2,150 training, 250 calibration, and 276 dev records. The separate 324-record Nimble evaluation set remains frozen test. Schema/question-order variants are generated from training data only; validation and test are not augmented. These records are synthetic/model-checked, and the extension is not Cloudflare's internal dataset.

**Training recipe.** The launcher defaults to the head-only stage and trains for up to 20 epochs, with early stopping after 3 stale epochs based on inner-dev mean record loss. It uses micro-batch 1, gradient accumulation 8, max length 4,096, head learning rate `1e-4`, seed 17, label smoothing 0.05, Brier weight 0.1, and gradient clipping at 1. The optional LoRA stage uses rank 256 and learning rate `2e-5`; its explicit attention, MLP, and linear-attention target modules are local choices, not Cloudflare-published settings. See [method card](retrain/15_clef_style_4b/METHOD.md), [model config](retrain/15_clef_style_4b/configs/model.yaml), [supervised config](retrain/15_clef_style_4b/configs/sft_r256.yaml), and [trainer](retrain/15_clef_style_4b/src/train.py).

**Status and caveat.** The supervised path is registered as runnable, but no clean training result is claimed. The initial local head shape check is not a training result. The current host's 11 GB GPU and CUDA/runtime mismatch do not support the planned 4B training setup; run on compatible higher-memory CUDA hardware and record actual peaks. RLCD remains deferred; any future local version must be labeled RLCD-like because Cloudflare's exact objective is unpublished.

### M16 — AdaS1 layerwise probe (P0 only)

**Method.** P0 captures frozen Qwen3.5-4B representations at layers 4, 8, 12, 16, 20, 24, 28, and 32 using M15's state/schema serialization. It trains identical small question-option probes at each depth and compares accuracy, NLL, Brier, and ECE. This is a representation-quality screen; it does not implement dynamic layer skipping or establish a latency gain.

**Revalidation.** Recent prior art already includes OD-1's layer-8 self-exit and RSI-Jev v5.0-VL's static 20-layer Qwen3.5-4B-Base model. M16 therefore treats static truncation as an existing baseline and focuses any later contribution on risk-calibrated multi-exit control, if supported by evidence. The pasted proposal's use of the 324-row outer set to choose exits was corrected: M16 P0 uses M10's family-disjoint 2,268/408 train/dev split and never opens outer test during probe or exit selection.

**Status and caveat.** `python3 scripts/run_training_methods.py M16 --stage probe` is the only runnable M16 stage. It needs a compatible CUDA environment and sufficient memory for BF16 hidden-state capture. The current `.venv` has PyTorch 2.14.1+cu130 and Transformers 5.18.0, but CUDA initialization fails against the installed driver; the exposed GTX 1080 Ti has 11 GB. Teacher training, multi-exit distillation, answerability, gate, staged serving/batch compaction, LoRA, and VLM are deferred pending P0.

The [M16 readiness audit](retrain/16_adas1_multiexit_4b/PROBLEMS.md) confirms P0 data and implementation readiness, while recording that the configured VKong output volume is deleting and the target CUDA stack has not passed a GPU smoke run.

## Published comparison recipes in `released/`

These entries help distinguish local experiments from published or frozen references. The frozen baselines below have no training data or optimization stage. The trained releases are already-published artifacts, not additional locally runnable methods in `scripts/run_training_methods.py`.

| Reference | Motivation and method | Training data and recipe represented in this workspace |
|---|---|---|
| [SemIf / OpenJev](released/00_semif/METHOD.md) | Zero-training direct option-token logits; establishes a frozen-model decision baseline. | No training and no training data. |
| [Simple Jev](released/01_simple_jev/METHOD.md) | Typed classifier serving interface using next-token scoring. | Default serving requires no training. Upstream has an optional RFDT workflow, which is separate from the workspace baseline. |
| [Tev1-4B](released/02_tev1_4b/METHOD.md) | LoRA SFT to emit one answer letter; comparison with direct candidate probabilities. | Published new-v1 data: 37,840 train and 4,568 dev across public classification datasets and generated policy/routing tasks. Recipe card records rank 8, alpha 16, LR `5e-5`, batch 8, sequence length 2,048, one epoch, seed 42; source licenses and label origins vary. |
| [JevK5-4B](released/03_jevk5_4b/METHOD.md) | Teacher distillation plus replay, then option-logit scoring at inference. | v0.3 card reports 17,408 teacher questions plus 30,052 human-labeled replay items (47,460 total); v0.1/v0.2 hyperparameters are not asserted as the v0.3 recipe. |
| [Kev-4B](released/04_kev_4b/METHOD.md) | LoRA plus a pointer head for candidate probabilities. | decision-v7 (10,000 examples), generated policies/rules, plus later skills delta and replay. Published base recipe reports rank 16, LR `5e-5`, two epochs; delta is a separate later update. |
| [Clef-Flash 9B](released/05_clef_flash_9b/METHOD.md) | Joint schema head with frozen Qwen backbone and rank-256 adapters; typed multi-field probabilities. | Cloudflare describes internal synthetic data with field, prompt, and schema permutations, followed by label-smoothed CE + Brier and RLCD. Exact sources, split, coefficients, adapter targets, and RLCD objective are not public, so exact reproduction is unavailable. |

## Running the local registry

From the workspace root, `python3 scripts/run_training_methods.py --list` shows registered and deferred methods. Train M10–M15 with `python3 scripts/run_training_methods.py M10` (replace M10 with the desired method), or run the six registered methods serially with `python3 scripts/run_training_methods.py all`. M14 is registered, but its preflight currently blocks it until licensed image-decision data and a compatible CUDA runtime are ready. For M15's optional LoRA stage, pass `--clef-stage lora`; default is `head_only`. Run AdaS1 P0 separately with `python3 scripts/run_training_methods.py M16 --stage probe`. Each launcher run creates a report under `results/training_runs/<run-id>/`. See [the runbook](TRAINING_RUNBOOK.md) for evaluation scope.
