# VLM System-One 4B

## Status

**V1 implemented; launch blocked on dataset and compatible CUDA runtime.** The full user-supplied methodology is preserved in [METHODOLOGY_PROPOSAL.md](METHODOLOGY_PROPOSAL.md). No M14 training run or benchmark result is claimed.

## Reconciliation with existing methods

This is a multimodal extension, not a duplicate of one existing recipe:

| Existing method | Shared idea | Difference for M14 |
|---|---|---|
| [M10 Nimble](../10_nimble_4b/METHOD.md) | Qwen3.5-4B, candidate decisions, non-generative inference | M10 scores allowed answer-token logits from text; M14 scores each image-conditioned candidate with a shared scalar readout. |
| [M11 CLM-style](../11_clm_style_4b/METHOD.md) | Candidate/decision representations and a learned readout | M11 is a text-only contrastive objective; M14 uses image + state + question + candidate and listwise choice probabilities. |
| [M12 pointer head](../12_pointer_head_only_4b/METHOD.md) | Frozen-backbone head-only adaptation | M12 uses Kev's text pointer representation; M14 tests Qwen's multimodal hidden state and a scalar candidate head. |
| [M13 full FT](../13_kev_fullft_4b/METHOD.md) | Adaptation-depth upper bound | M13 is a text pointer scorer; M14's V stages progressively unfreeze language/vision components. |
| [M15 Clef-style](../15_clef_style_4b/METHOD.md) | Qwen3.5, non-generative decisions, CE + Brier, LoRA | M15 routes mixed text schemas through a joint schema head. M14 is image-grounded candidate scoring and adds modality reliance, image-group leakage, and text-retention evaluation. |

The clearest incremental questions are **text-to-image decision transfer**, **minimum adaptation depth**, and **visual reliance/calibration under modality shift**. “Multimodal Qwen with calibrated probabilities” alone is not a distinct contribution; the proposal itself notes prior OneJev/OmniJev overlap. Cite and compare those prior methods before making novelty claims.

## Objective and interface

Given one or more images, state, question, and typed candidates, return a probability per candidate without decoding answer text. Each candidate is scored independently with the same function:

```text
(images, state, question, one candidate)
  → Qwen3.5 multimodal backbone
  → final non-padding language hidden state
  → shared linear scalar head
  → candidate score
```

For a question with candidate scores `s`, compute `softmax(s / T)` across that question's candidates. Fit `T` only on the dedicated calibration split. Candidate IDs are metadata and are excluded from model text unless semantically meaningful. Group losses by question so questions with different candidate counts have equal weight. The candidate branch is permutation-equivariant only when its serialized inputs are identical apart from candidate content; test that property rather than assuming it.

Choice, yes/no, and ordered score are represented as candidate sets. Abstention is an explicit candidate only when supported by labels. Do not synthesize no-evidence labels or counterfactual grounding pairs without source evidence.

## Model and transfer decision

- Primary checkpoint: `Qwen/Qwen3.5-4B`, immutable workspace revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Base checkpoint: optional ablation only, `Qwen/Qwen3.5-4B-Base`, revision `1001bb4d826a52d1f399e183466143f4da7b741b`.
- The post-trained model is the primary starting point, consistent with the proposal. Keep the model's vision tower and processor from the same pinned multimodal checkpoint.
- **V1 has no transfer dependency:** it is a fresh linear-head baseline. A separate transfer arm still requires a validated text winner and an exactly compatible adapter, head, and prompt template.
- The trainer uses the Qwen3.5 conditional-generation model's multimodal backbone and the processor's chat template to obtain final hidden states without materializing vocabulary logits. Transformers 5.18 exposes these model classes, and the pinned processor successfully converted a local test image into multimodal tensors. A full model forward/training pass has not been exercised because CUDA is incompatible on this host.

## Dataset and splits

No dataset is selected for M14. The workspace's Flowers-102 records are a useful vision benchmark shape, but their source lock says `VERIFY_UPSTREAM_TERMS`; they cannot be treated as cleared training data. They also cover only one fine-grained classification task, not the proposed broad mixture. The manifest remains `not_ready` until sources, image terms, provenance, target construction, group splits, and checksums are documented.

Required first controlled release:

- Start with a small, auditable, licensed image-decision dataset or mixture. Record image rights separately from dataset-code license; do not use the proposed 100K–300K target as a launch prerequisite.
- Store one normalized example per question with stable `id`, `source`, `source_family`, `group_id` (image/video/episode), `image` path(s), state, question, typed candidates, and gold candidate ID.
- Keep all records sharing an image or source group in one partition. Freeze train, model-validation, calibration, and test IDs; keep OOD datasets separate. Only train/dev/calibration can be read by the trainer/calibrator respectively.
- Hash source manifests and split files. Exact and perceptual image deduplication plus text/group leakage checks are prerequisites before training.
- Add text-only, correct-image, shuffled-image, and image-only evaluation conditions on image-essential records. Test performance should demonstrate image reliance, not just multimodal input acceptance.

See [DATASET.md](data/DATASET.md) and [manifest.json](data/manifests/manifest.json) for the current gate.

## Stages and decision rules

1. **V0 (no training, proposed):** compare answer-token, candidate sequence-likelihood, and generative baselines. This stage is not included in the M14 trainer.
2. **V1 (implemented first trained arm):** freeze vision, merger, and language backbone; train one fresh linear scalar candidate head. The local config uses CE + Brier (`lambda_brier=1.0`) and selects by validation `NLL + 0.25 * Brier`. It logs progress for every question update and saves resumable state. The current CLI does not yet implement the proposed Brier-weight ablation sweep.
3. **V2:** frozen vision/merger + language LoRA + head. This is the first adaptation arm after V1. Start rank 32, alpha 64, dropout 0; use explicit language-only targets and audit trainable module names to exclude vision.
4. **V3:** V2 plus vision merger, only if V2 has strong text retention but weak image-essential/visual-OOD results.
5. **V4:** V3 plus top four vision blocks, only if V3 evidence indicates visual representation is the bottleneck.
6. **V5:** full VLM upper bound only on suitable multi-GPU memory, after V1–V3 validate data/model/reload paths.

Run V stages sequentially with fresh outputs. Fit temperature on calibration data after model selection. Keep test and OOD records inaccessible to training and checkpoint selection. Preserve a text replay arm only if there are compatible text records and their source licenses permit it; report text retention separately. Grounding margin loss remains off until verified counterfactual image pairs exist.

## Evaluation

Report per-question accuracy, NLL, Brier, ECE, AURC/selective risk, candidate-order sensitivity, source/domain OOD, visual-corruption robustness, text retention, and latency/peak VRAM. On image-essential evaluation slices, compare full, text-only, image-shuffled, and image-only conditions. Use paired confidence intervals for stage comparisons. Do not use the project's ordinary text methods as direct score rankings unless records, splits, and task format are matched.

## Readiness and known limits

- Model artifacts are already present and pinned under `shared/models/`; M14 must record the same immutable SHA in its lock.
- This host has one GTX 1080 Ti with 11 GB VRAM. V1 requires at least 23 GiB runtime-visible GPU memory; 24 GB-class cards may report less than 24 GiB to PyTorch. Record actual peak memory during the pilot.
- The workspace `.venv` has PyTorch 2.14.1+cu130 and Transformers 5.18.0, but CUDA is unavailable because the installed NVIDIA driver reports CUDA 12.2. Pillow 12.3.0 and torchvision 0.29.1 are installed and declared in base requirements. System `python3` has no PyTorch.
- V1 training, prediction, strict split validation, and generic metrics are implemented. V2 LoRA, V3/V4 vision unfreezing, V5 full fine-tuning, temperature fitting, and modality-shift evaluation remain unimplemented.
- No image-decision data is selected or rights-cleared. The generic runner registers M14, but readiness preflight rejects the current `not_ready` manifest before training.
- Dataset-scale suggestions in the proposal (100K–300K, mixture percentages, pixel budgets, hyperparameter ranges, VRAM estimates) are hypotheses/planning values, not verified requirements.

## Provenance

Base repo: `Qwen/Qwen3.5-4B`; revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` from `shared/models/MODELS.lock`. Dataset: none selected. Full proposal: [METHODOLOGY_PROPOSAL.md](METHODOLOGY_PROPOSAL.md).
