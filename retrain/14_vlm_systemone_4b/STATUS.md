# Status: V1 IMPLEMENTED — LAUNCH BLOCKED BY DATA AND RUNTIME

The fresh-head V1 training loop, prediction adapter, split/provenance audit, resumable checkpoints, step-level JSON progress, and inner-dev metric integration are implemented. The generic training launcher now registers M14. This makes the experiment executable once its hard launch gates pass; it does not mean a training run has been completed.

Training remains blocked because:

1. No M14 image-decision source has verified image-use terms, source provenance, populated group-safe splits, or checksums. The existing vision benchmark locks still say `VERIFY_UPSTREAM_TERMS`; the trainer refuses the current `not_ready` manifest.
2. The workspace `.venv` has PyTorch 2.14.1 built for CUDA 13 while the installed driver reports CUDA 12.2, so PyTorch exposes no usable CUDA device. Pillow 12.3.0 and torchvision 0.29.1 are installed and the pinned processor successfully produced image-conditioned tensors; only a full model forward/training pass remains unverified on this host.
3. V1 enforces a 23 GiB runtime-visible planning gate so 24 GB-class 3090/4090 cards that report about 23.5 GiB pass preflight. Actual peak must still be measured in the pilot. M14 LoRA and later unfreezing stages are not implemented by this V1 trainer.

The text winner is not a V1 blocker: V1 is clearly labeled as a fresh-head baseline. It remains a dependency for the separate transfer experiment.

See [PROBLEMS.md](PROBLEMS.md) for the full list of data, runtime, implementation, evaluation, and research gaps.

After obtaining licensed data and a compatible CUDA environment, populate the files and attestations described in [DATASET.md](data/DATASET.md), run the full audit, then launch with:

```bash
python3 scripts/run_training_methods.py M14
```

Progress is saved to `retrain/14_vlm_systemone_4b/results/training_runs/<run-id>/M14/progress.jsonl`; `last.pt` supports resume and `best.pt` is selected by inner-dev NLL + 0.25 × Brier. No training was started during this setup because the data and runtime gates fail on this host.
