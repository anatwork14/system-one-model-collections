# Adaptation notes

This directory is an adaptation of OpenJev-RLCD, not a claim to reproduce TypeSafe Jev.
The pristine upstream repository is preserved under `upstream/` at the commit in
`upstream_commit.txt`; the adaptation lives in this directory.

Planned compatibility adaptations:

- Load the official Qwen3.5-4B text configuration in BF16 and omit unused vision weights.
- Train discovered Qwen3.5 text projection LoRA modules while keeping the base frozen.
- Share one frozen base model between active and reference LoRA adapters. The Stage-2
  reference adapter is copied from the Stage-1 checkpoint and never optimized.
- Keep the source `single_surrogate` / lambda=0 proper-score estimator and KL shaping;
  Stage 1 detaches rationale log-probabilities, while Stage 2 enables the score-function
  term at `sf_coef=0.3` and retains pathwise gradients.
- Use atomic adapter + optimizer/RNG checkpoints to support exact continuation.

No GPU behavior, training result, or performance result is claimed until the VKong
architecture and smoke gates pass.
