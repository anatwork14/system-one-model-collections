# M17 — Execute OpenJev-RLCD on Qwen3.5-4B using VKong

## Role and execution mandate

You are the senior ML research engineer implementing **M17-A: an adapted, reproducible OpenJev-RLCD training experiment** within our LLM-derived Decision Model training-method collection. Do real code inspection, implementation, tests, training and evaluation. Do not respond with only a plan. Report completed steps and blockers truthfully; do not fabricate GPU timings, successful runs, or results.

**Source repository:** https://github.com/ZimmyGao/openjev-rlcd

**Source commit examined while preparing this brief:** `22db40ad3333f5cf0bc5139097da70e0f8c05018` (2026-10-01). Verify the commit and pin the revision actually used. Preserve the original repository as a reference (git submodule/vendor copy or separate clone); create our own versioned adaptation and record its diff.

**Target:** `Qwen/Qwen3.5-4B` with its exact Hugging Face revision pinned. Task: `gsm8kv` (GSM8K-Verify). Run one seed, 17 for Stage 1 and 1017 for Stage 2, and clearly distinguish these training seeds from the fixed dataset-split seed.

**GPU provisioning:** Use **VKong** as the exclusive GPU provider/runner. Discover the actual VKong interface, credentials integration and job APIs/CLI from our existing project/environment; do not invent VKong commands or bypass VKong by directly provisioning with Vast.ai or RunPod. Target **one RTX 6000 Ada 48GB or equivalent-or-better GPU**. Prefer a persistent storage volume, ≥100GB working disk, sufficient host RAM (prefer ≥64GB), CUDA compatible with the selected PyTorch build, and remote log/checkpoint upload. Use the least costly viable offer. Track real GPU-hours/cost. Do not embed credentials in code or logs.

**Compute policy:** Max **8 GPU-hours of training**, combined across Stage 1 and Stage 2, excluding separate evaluation if provided another allowance. Optimize toward the original `400 + 400` steps; do not promise this is achievable before measuring. Enforce a safety ceiling and an early checkpoint if runtime exceeds projection. Include smoke/pilot training time in the 8 hours. Use measured GPU runtime, not wall-clock time for queue/provision/download, when reporting training hours. No unbounded spending or retries.

## Background / scientific invariants

The paper is *OpenJev-RLCD: A Working RLCD Implementation* (https://arxiv.org/abs/2609.38850), an independent implementation; **do not claim this reproduces TypeSafe Jev's proprietary training**. We are adapting the paper from Qwen3-1.7B to Qwen3.5-4B (and possibly LoRA); label outputs `adapted_openjev_rlcd_qwen35_4b`, **not exact reproduction**.

For each example x, sample M on-policy rationales r_i. At the `Answer:` boundary, **READ** the model's probability distribution over valid answer candidates: `u_i = softmax(logits[valid_label_token_ids])`. Do not sample an answer to train the RLCD reward and do not replace it with a generated text answer. Score **each rationale individually** with `J(u_i,y) = 2u_i[y] - sum_k(u_i[k]^2)` (Brier-derived strictly proper score, up to label-independent constants). `diversity_weight = 0`, not mixture-scoring (`lambda = 1`).

- **Stage 1 (`calibrate`):** sample model rationales, train **pathwise gradient through u_i**, but **do not apply the score-function/policy-gradient update to the sampled rationale log-probabilities**. In the original implementation, parameters throughout the model can still update via the pathwise loss; this is NOT necessarily a frozen backbone or classifier-head-only training. The condition is that the REINFORCE path is disabled.
- **Stage 2 (`reinforce`):** initialize from the Stage-1 checkpoint; apply the same per-rationale proper score with **REINFORCE on rationale log-probabilities**, leave-one-out baseline, `sf_coef = 0.3`, `kl_beta = 0.04` against a **frozen Stage-1 reference**. Preserve the pathwise gradient term. A fresh optimizer and new training seed are expected. Do not silently substitute GRPO or ordinary reward-on-argmax.
- Evaluate **single-query** output (`u_1`) as the primary result, optionally a 4-rationale mixture as a secondary comparison. Reproduce separate temperature scaling learned only on dev/calibration data.

Do not make mathematics change unintentionally while adding LoRA, quantization, caching, microbatching or a shared model. Preserve the gradient pathways, reference-KL semantics and on-policy sampling behavior. If the exact objective cannot be preserved, document the divergence and prove/measure the consequences.

## 1. Inspect repository and establish provenance

1. Clone and pin source SHA; inspect `README.md`, `requirements.txt`, `scripts/common.sh`, `scripts/two_stage.sh`, `scripts/smoke_test.sh`, `openjev_rlcd/train.py`, `openjev_rlcd/estimators.py`, `openjev_rlcd/tasks/__init__.py`, `openjev_rlcd/tasks/gsm8k_verify.py`, `openjev_rlcd/metrics.py`, `openjev_rlcd/evaluation.py`, and `tests/`.
2. Note original Qwen3-1.7B trainer has an FP32 trainable model, a BF16 generation copy, an optional BF16 reference copy, full AdamW optimizer, `.model`/`.lm_head` assumptions, and full-model state checkpoints. These cannot be assumed to fit 4B/48GB unchanged.
3. Record environment and dependency versions, commit hashes, checkpoint revision, dataset JSON SHA256, GPU model/driver, available VRAM, CPU RAM, CUDA/PyTorch/Transformers/PEFT versions.
4. First run CPU estimator unit tests exactly as supplied: `python -m pytest tests`. Do not edit estimators before establishing the original tests pass.
5. `scripts/two_stage.sh` launches **five runs per seed** (Stage 1 plus four Stage-2 forks), and defaults to **three seeds**. **Do not run it directly** under an eight-hour budget. Implement a **single-pair launcher** for Stage 1 followed by one RLCD Stage 2.

Reference setup (run inside the VKong job after repository clone):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Baseline pinned research dependencies; check GPU/Python compatibility before installing:
pip install -r requirements.txt
python -m pytest tests
python -m openjev_rlcd.train --help
```

For the Qwen3.5 adapter install a verified compatible `transformers`, `peft`, `accelerate`, and optionally `bitsandbytes`, `safetensors`, etc. **Do not blindly upgrade the original reproducibility environment.** Put adaptation dependencies in a separate pinned requirements/lock file, with a successful import/forward/backward test. Prefer official Hugging Face checkpoint `Qwen/Qwen3.5-4B`, not an unnamed unofficial conversion. If using the official model through a text-only class and omitting vision weights, verify that the tokenizer and text logits match the original; document exactly what was loaded and trainable.

## 2. Validate the Qwen3.5-4B architecture + tokenizer BEFORE spending training budget

1. Verify correct Qwen3.5 model class and whether `AutoModelForCausalLM` is supported by our pinned Transformers release. The official checkpoint is multimodal; **do not assume `.model` and `.lm_head` paths from Qwen3-1.7B still work**. If needed, instantiate `Qwen3_5ForCausalLM` from the official checkpoint and adapt path access. Avoid `trust_remote_code=True` unless explicitly required and reviewed.
2. Verify differentiable forward computation through hybrid Gated DeltaNet/attention modules in BF16 using checkpointing. Detect incompatible/missing fast-kernel dependencies; benchmark actual kernels instead of trusting theoretical FLOPs.
3. Disable Qwen3.5 native `thinking` template feature for this experiment **only if the tokenizer supports it and the rendered prompt is verified**. The task still explicitly asks the model to generate an observable rationale terminated by `Answer:`. Do not accidentally produce model-internal `<think>` sections that confuse rationale extraction. Inspect at least 20 generated examples.
4. Verify `" Answer:"` marker/end boundary across generation and forward replay, including stop-string behavior, EOS/truncation and forced insertion of `\nAnswer:`. `forced_marker_rate` and length-limit rate must be logged.
5. Fix/check tokenization of label strings `" yes"`, `" no"` (and later MMLU letters). The upstream initializer takes only the first token of `tok.encode(...)` without checking length. Require **exactly one token per label** and uniqueness, or implement explicit multi-token sequence scoring with matching semantics. Never silently use truncated token IDs. Verify candidate probabilities sum to 1, and manually inspect readout logits at the exact answer boundary.
6. Numerical safety: no NaN/Inf in token log-probs, scores, rewards, gradients or loss. Check padding/mask/position interactions on Qwen3.5's hybrid architecture. Verify `logp_r` sums only rationale tokens and excludes prompt/forced marker tokens as intended.

## 3. Dataset and split isolation

Primary dataset is the repo's released `data/gsm8k_verify.json`: GSM8K problems, proposed numeric answers generated by Qwen3-0.6B, and exact-correctness binary labels. Do **not** rebuild it unless absent or corrupted. `openjev_rlcd/tasks/gsm8k_verify.py` defaults to `n_train=4000`, `n_dev=500`, `split_seed=0`; GSM8K original `test` problems are held out. `--dev-cap 300` limits evaluation workload; `--test-cap 1260` is the original script's requested cap, but evaluate actual held-out count present.

- Validate JSON schema, correct/incorrect balance, source, hashes, duplicate IDs, split assignment and cross-split question overlap.
- Pin the exact source file and split-seed; record train/dev/test UIDs. No data augmentation, training, supervised temperature fit, or checkpoint selection using the held-out test set.
- Make calibration/dev evaluation separate from final test evaluation. Fit post-hoc temperature scaling using held-out dev/calibration labels only; clearly state if temperature calibration and model selection use the same dev pool. Prefer a stable fixed partition if feasible.
- The training sampler samples items with replacement, so 400 optimizer steps × 8 items/step does **not** guarantee seeing all 4,000 unique examples. Log seen unique UIDs.
- Preserve the shared study's Nimble 324-row outer test set untouched; this M17-A task uses GSM8K-Verify only. Transferring RLCD to M15 is a separate later research question.

## 4. Implement memory-efficient RLCD (48GB)

Implement a minimum invasive **LoRA adaptation** (or memory-equivalent method) with tests. Suggested starting point: BF16 language backbone, trainable LoRA adapters (rank 16 or 32 initially, not treated as original-paper settings), gradient checkpointing, microbatching, optimizer over trainable adapters only, limited concurrent generations and optionally 8-bit optimizer/offloaded reference if necessary. Track actual trainable parameter count and peak memory. Model-specific LoRA targets must be discovered from `named_modules()`; do not copy Qwen3-1.7B module names blindly. Avoid training vision parameters in this text task.

Memory must accommodate trainable model, gradients/optimizer, on-policy sampling and immutable Stage-1 reference. Options include synchronizing the generation adapter from the training adapter, temporarily switching adapters to a frozen reference for KL, CPU offload or quantized reference **only when outputs and KL semantics have been checked**. Do not keep three full independent 4B weights on GPU if they breach 48GB. Generation and gradient forward must use equivalent current policy weights at each step; ref forward uses **Stage-1 frozen weights** in Stage 2.

Keep the original `lambda_surrogate`, `single_surrogate`, leave-one-out variance reduction and reference regularization mathematically unchanged unless a well-tested necessary patch is documented. Stage 1 must truly disable rationale score-function gradients. Stage 2 must show nonzero, finite pathwise AND score-function contribution; KL reference frozen; gradients reach intended adapter weights. Unit test frozen-reference invariance, Stage-1 checkpoint initialization, correct score, score-function detach, `sf_coef`, `kl_beta`, and zero-variance Brier examples.

Support **atomic/resumable checkpoints** saving LoRA adapter weights, optimizer/scheduler state, RNG states, step, elapsed GPU time, SHA manifests, dataset split, all hyperparameters and calibration/eval settings. An adapter-only checkpoint without optimizer/RNG must be described as restart-from-weights, not exact resume. Upload checkpoint/logs durably before GPU termination.

## 5. Eight-hour training schedule and decision gates

**Primary goal:** one seed, Stage 1 (400 steps) + Stage 2 (400 steps) as close as possible to paper settings: `items_per_step=8`, `m=4`, `max_new_tokens=320` for gsm8kv, `diversity_weight=0`, `kl_beta=0.04`, `sf_coef=0.3` in Stage 2; dev/test caps `300/1260`; `eval_every` set to avoid long in-training evaluation. Original optimizer LR defaults to `2e-6`; LoRA LR must be tuned cautiously on dev, not copied blindly. Freeze all experiment settings after the smoke phase.

**Safeguards:** Provision GPU through VKong only. Start with 2+2 step end-to-end smoke test, then 10–20 timed warm steps per stage with the *actual* generation length, m, batch and memory configuration. Include pilot steps in training-hour accounting. Time GPU synchronize-aware p50/p90 iteration durations, prefill/decode time, KL forward, peak VRAM. Predict cost of remaining Stage-1/Stage-2 steps, adding margin for IO/checkpoint overhead.

**Adapt steps to measured throughput.** Eight hours for 800 steps implies average **≤36 seconds/step** across both stages, including training evaluation/checkpoint overhead. Preferred operational target **≤30 seconds/step**. If infeasible, do not pretend a completed 400+400 run; choose an explicit matched smaller budget (e.g. 200+200, or measured split), preserve both stages, save progress, and report achieved steps/tokens/GPU hours. Allocate budget only after verifying stages are stable; the second stage must get a meaningful nonzero allocation. Do not lower `m` below 2 (requires at least 2 for per-rationale leave-one-out); using m=4 is preferred.

Use an absolute 8 GPU-hour guard to checkpoint and stop safely (avoid termination mid-write); include up to 5–10 minutes safety buffer. Detect NaN/Inf, OOM, unbounded loss, KL explosion, missing `Answer:` rate, zero rewards/gradients or catastrophic reasoning collapse; act conservatively, save a diagnostic snapshot and stop rather than looping endlessly. **No three-seed run, no entire `two_stage.sh`, no GRPO fork** in the first budget.

## 6. Minimal intended run semantics (adapted CLI, not automatically executable upstream)

The following is a **behavioral contract for a new launcher**, not proof that the upstream CLI can run 4B with LoRA unchanged. Implement `scripts/run_m17_qwen35_4b.sh` and `configs/m17_qwen35_4b.yaml` whose actual flags match implemented code. These original argument names MUST remain meaningful when possible:

```bash
# Stage 1: on-policy rationale sampling; pathwise proper-score ONLY
python -m openjev_rlcd.train \
  --model Qwen/Qwen3.5-4B --task gsm8kv \
  --m 4 --m-eval 4 --max-new-tokens 320 --items-per-step 8 \
  --steps 400 --dev-cap 300 --test-cap 1260 \
  --diversity-weight 0 --kl-beta 0.04 --no-score-function \
  --seed 17 --save-checkpoint runs/m17/stage1.pt \
  --output runs/m17/stage1.json

# Stage 2: pathwise + rationale policy gradient, frozen Stage-1 reference
python -m openjev_rlcd.train \
  --model Qwen/Qwen3.5-4B --task gsm8kv \
  --m 4 --m-eval 4 --max-new-tokens 320 --items-per-step 8 \
  --steps 400 --dev-cap 300 --test-cap 1260 \
  --diversity-weight 0 --kl-beta 0.04 --sf-coef 0.3 \
  --init-checkpoint runs/m17/stage1.pt --seed 1017 \
  --save-checkpoint runs/m17/stage2.pt \
  --output runs/m17/stage2.json
```

**Warning:** These are upstream-style **semantic examples only**. The upstream loop must be modified for Qwen3.5 + LoRA, may need adapter-specific checkpoints, and currently saves full BF16 state dict. Do not run the quoted commands until compatibility and 48GB memory tests pass. The original Stage-1 config's `--kl-beta` does not imply actual reference KL gradients when `--no-score-function` is active; preserve behavior and document it.

## 7. Evaluation after training (not counted toward training budget unless GPU time is explicitly capped overall)

At minimum evaluate the unchanged base model, saved Stage-1 checkpoint and saved Stage-2 checkpoint on exactly the same held-out UIDs. If possible add SFT + temperature scaling as a separate later baseline. Evaluate **single-rationale probabilities** as primary; `mixture` M=4 secondary. Fit temperature on dev only and report both raw and post-TS metrics:

- Accuracy, Brier score, NLL, ECE; AURC (lower is better); coverage at predicted 5% error, **realized risk at that coverage**, and coverage/risk curves; optional JSD; mean rationale tokens, marker force/truncation rate and repetition rate.
- Report predicted risk separately from realized error. Never claim 5% realized error is guaranteed by predicted-confidence thresholding alone. Give uncertainty intervals if possible. Use per-item predictions and UIDs for paired comparisons.
- Compute training GPU-hours, elapsed time/stage, total rationale tokens, mean/p50/p90 step time, peak allocated/reserved VRAM, GPU utilization where measured, throughput and inference p50/p95 latency for one query (single rationale), and any cost data from VKong.
- Correctly disclose that a single seed is a **pilot**, not a paper-level replication of the source's three-seed statistical evaluation. Do not compare reported published 1.7B performance to 4B run as though controlled.

On failure, report diagnostics and last good checkpoint; do not substitute invented metrics. Hold the test data closed until training and config selection end.

## 8. Artifacts and source-control requirements

Create clean, versioned outputs (exact paths may integrate with our workspace):

```
experiments/m17_openjev_rlcd/
  README.md
  upstream_commit.txt
  PATCH_NOTES.md
  configs/m17_qwen35_4b.yaml
  requirements-adapter.lock
  scripts/run_m17_qwen35_4b.sh
  tests/test_qwen35_readout.py
  tests/test_rlcd_gradient_paths.py
  tests/test_stage1_stage2_state.py
  manifests/{model,dataset,environment}.json
  runs/<id>/stage1/{train.log,metrics.json,checkpoint...}
  runs/<id>/stage2/{train.log,metrics.json,checkpoint...}
  runs/<id>/eval/{base,stage1,stage2,comparison}.json
  runs/<id>/summary.md
```

Generate and preserve: commands and effective arguments, pinned SHAs, environment freeze, modifications versus upstream, dataset split UID hashes, actual steps completed, validation results, sample trajectories with sensitive data redacted, failure events, artifact URIs, and VKong job/GPU identifiers. Ensure logs and artifacts are downloadable/persistent after termination. Keep any produced model weights private unless explicitly authorized to release; respect Qwen/benchmark licenses. Do not commit secrets, full datasets or multi-GB checkpoints to git.

## 9. Acceptance criteria / final response format

Declare **TRAINING COMPLETED** only if: correct Stage-1 and Stage-2 semantics were unit-tested, data leakage checks passed, VKong run happened, both checkpoints were saved and verified loadable, actual GPU timings and loss curves exist, and a held-out evaluation report exists. If only one stage or partial steps finish, say **PARTIAL**.

In your final report, include:
1. Which GitHub commits / model revisions / dataset SHA / VKong GPU were used.
2. Which original features are unchanged and every adaptation to 4B, LoRA, memory, kernels and prompting.
3. Stage 1 vs Stage 2 exact training configurations and achieved step count, training-hours and cost.
4. Numerical objective/gradient validation results, memory/timing checks and failures.
5. Base vs Stage 1 vs Stage 2 table: accuracy, Brier, NLL, ECE, AURC, Cov@5% and realized risk, after proper dev-only TS.
6. Paths/links to configs, logs, checkpoints and evaluation artifacts.
7. What remains unverified (especially one-seed limitation, native Jev equivalence and cross-method comparison).

**Execution priority:** get one scientifically defensible, reproducible two-stage result, not the largest possible step count. Start implementation immediately and proceed through tests, VKong provisioning, smoke, timed pilot, full available training, and evaluation, stopping only at concrete safety/blocker gates.
