# M16 — AdaS1: risk-calibrated adaptive depth

## Status

**P0 frozen-layer probes completed on a VKong RTX 4090 on 2026-10-06.** The verified archive and operational lessons are in [VKONG_RUNBOOK.md](VKONG_RUNBOOK.md). P0 is an architecture screen, not a trained adaptive model or evidence of early-exit speed. Only `--stage probe` is registered as runnable. Teacher, multi-exit distillation, answerability, gate, staged serving, LoRA, and VLM stages remain deferred; they require implementation and validation before another paid training run.

See [the readiness audit](PROBLEMS.md) for current code/data status, GPU and VKong blockers, and the launch sequence.

## Question and prior-art boundary

M16 asks whether a typed decision model can allocate Qwen3.5 decoder depth per request and improve the quality/latency frontier under matched conditions. Early exit, adaptive depth, and static truncation already exist. OD-1 reports a layer-8 self-exit that reached its full-depth validation accuracy only after escalating 99.7% of requests at confidence threshold 0.99. RSI-Jev v5.0-VL (released 2 October 2026) already runs Qwen3.5-4B-Base through layer 20 and reports a flattened quality curve at that point. These results make P0 a falsifiable local check, not a novelty claim.

The possible contribution, if later stages work, is narrower: risk-calibrated multi-exit control for runtime-defined typed decisions, with cross-exit stability, answerability supervision, and real staged execution. A formal prior-art review is still required before publication.

## P0 method

- Base: local `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Input/output interface: M15's state plus runtime-defined Choice/Noul/Score schema serialization and output contract.
- Data: M10's family-disjoint 2,268 train / 408 inner-dev split. P0 does not open the frozen outer 324 rows. The labels are synthetic/model-checked and not fully human reviewed.
- Frozen backbone: collect pooled question, option, and global-state vectors from decoder layers 4, 8, 12, 16, 20, 24, 28, and 32.
- Readout: train one identical small probe per layer, using the same initialization seed, projection/scorer shape, CE + Brier objective, and inner-dev NLL checkpoint selection.
- Metrics: question accuracy, NLL, multiclass Brier, and 15-bin top-label ECE. Save feature caches, probe checkpoints, training histories, `exit_curve.csv`, `exit_curve.svg`, hardware, peak allocated/reserved VRAM, and elapsed time below `results/probes/<run-id>/`.
- Purpose: identify whether intermediate representations support a follow-on multi-exit experiment. The frozen-backbone probe does not establish that end-to-end early stopping will be fast.

The preserved 324-row outer test is not used to choose layers or tune the method. If P0 passes, freeze the exit design using inner-dev evidence before a separately registered final evaluation.

## Run

```bash
python3 scripts/run_training_methods.py M16 --stage probe
```

Run on compatible CUDA hardware with sufficient memory for a BF16 Qwen3.5-4B forward pass with hidden-state capture. The probe records the GPU's free/total VRAM and validates fit with actual model loading and feature extraction; it does not apply a fixed 24 GiB gate. The pinned runtime requirements include the FLA CUDA extra and prebuilt causal-conv1d wheel checks. The verified VKong runtime was PyTorch 2.8.0+cu128 / CUDA 12.8 on an RTX 4090; see [VKONG_RUNBOOK.md](VKONG_RUNBOOK.md) for setup and recovery details.

The root [VKong task config](../../vkong.yaml) validates with one RTX 4090, the PyTorch 2.8.0 CUDA 12.8 runtime, and no persistent storage or hourly cap. The completed P0 run used the existing rental, recovered its archive through the service endpoint, verified checksums, and stopped the app. Do not launch later M16 stages until they are implemented; full AdaS1 stages remain deferred.

## Deferred stages

1. Select up to three intermediate exits from P0 inner-dev curves.
2. Train a full-depth schema-head teacher using M15 serialization.
3. Distill intermediate heads with CE + Brier + teacher KL; add RPS for ordinal Score questions.
4. Test static exits, max-probability, entropy/margin, logistic gate, and learned risk gate.
5. Add explicit answerability/UNKNOWN only with an auditable open-set dataset and a split policy.
6. Implement true staged layer execution and batch compaction before making speed claims.
7. Consider rank-16 LoRA and VLM only after the frozen-backbone adaptive path clears quality and wall-clock kill tests.

## Known limitations

- P0 uses one small synthetic/model-checked source and the M10 serialization, so it is only an architecture screen.
- The current probe uses pooled spans and is intentionally smaller than the M15 joint schema head; later comparisons must hold the decision readout constant.
- P0 performs a complete 32-layer pass to capture all exits; it measures representation quality, not inference savings.
- Qwen3.5's hybrid linear-attention path depends on compatible Transformers and optimized `flash-linear-attention` / `causal-conv1d` kernels for meaningful future latency work.
