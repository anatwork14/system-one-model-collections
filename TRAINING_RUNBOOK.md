# Training and standard evaluation runbook

Use this runbook for the local training methods in this repository. Commands
run from the workspace root (`/mnt/ssd/genos2/workspace`). The launcher runs a
method's existing training implementation, evaluates its checkpoint on that
method's family-disjoint inner-dev split, and saves one standard report bundle.

## Commands

List implemented and deferred methods without starting training:

```bash
python3 scripts/run_training_methods.py --list
```

Train and evaluate one method by entering its ID:

```bash
python3 scripts/run_training_methods.py M10
```

Valid training IDs are `M10`, `M11`, `M12`, `M13`, `M14`, and supervised `M15`. `M16` currently supports only its P0 frozen layer-probe stage. M14 is registered but its preflight currently blocks launch until licensed image data and a compatible CUDA environment are present.
Train and evaluate the six registered methods serially with:

```bash
python3 scripts/run_training_methods.py all
```

For M15's supervised LoRA variant, use:

```bash
python3 scripts/run_training_methods.py M15 --clef-stage lora
```

M14 V1 uses a fresh candidate-scoring head over the frozen Qwen3.5 multimodal model. Before dispatch, prepare all five split manifests and images and pass the full leakage audit described in `retrain/14_vlm_systemone_4b/data/DATASET.md`. The trainer opens only train and model-validation labels; calibration, test, and OOD are evaluated separately. Step-level loss, ETA, and VRAM are saved in `progress.jsonl`; `last.pt` supports exact-settings resume. To resume through the standard launcher, supply the prior run directory:

```bash
python3 scripts/run_training_methods.py M14 --m14-output results/training_runs/<prior-run-id>/M14 --m14-resume
```

For AdaS1's first layer-probe experiment, run on a compatible CUDA host:

```bash
python3 scripts/run_training_methods.py M16 --stage probe
```

M16 uses M10's 2,268/408 family-disjoint train/dev split. Its probe script does not read the frozen 324-row outer test. It fits identical probes at decoder layers 4/8/12/16/20/24/28/32 and writes `exit_curve.csv`, `exit_curve.svg`, reports, feature caches, and checkpoints under `retrain/16_adas1_multiexit_4b/results/probes/`. Teacher, exit distillation, open-world, learned gate, dynamic serving, LoRA, and VLM stages are deferred until P0 results justify them. The current CUDA runtime cannot initialize against the installed NVIDIA driver, so run this on a compatible CUDA host.

To rerun evaluation from an existing expected checkpoint without training:

```bash
python3 scripts/run_training_methods.py M10 --skip-train
```

Replace `M10` with the desired method ID. The launcher prefers the workspace
`.venv` for its subprocesses when that environment exists. Training scripts'
`CONFIRM_TRAIN=yes` guard is set internally only after the user invokes the
launcher. A PyTorch CUDA preflight runs before each method; if CUDA is missing
or unusable, the method fails before training starts and the report records the
reason.

## Outputs

Every invocation gets a unique directory:

```text
results/training_runs/<run-id>/
  standard_report.json
  logs/
  native/<method>/
```

`standard_report.json` contains each method's status, checkpoint, metric JSON,
latency summary where measured, artifact paths, and a per-axis metric coverage
map. If `all` has a method failure, later methods still run and the report
ends with `completed_with_failures`; check that method's log before retrying.

## What the report means

M10–M15 use method-specific family-disjoint inner-dev data. Their scores are
useful for checking each training implementation and its probability outputs;
they are not a direct cross-method ranking because the evaluated records and
formats differ. M10–M13 native outputs are normalized by
`evaluation/standardize_training_eval.py`; M15 emits complete normalized
per-question records through its checkpoint inference adapter. The shared
classification, calibration, and selective-decision metrics are computed only
when the method provides labeled probability distributions.

The survey metrics for routing, open-set, open-world, end-to-end system
quality, cost, compute, local-serving, multi-objective, and robustness require
separate routing outcomes, benchmark partitions, serving traces, cost data,
continual-learning matrices, or paired perturbation runs. This training
launcher does not create those inputs. It marks those metrics unavailable in
the report rather than substituting model accuracy or zeros. M13 records
native request median and p95 latency; p99 and router overhead are absent.
M10 has no native inference latency. M11, M12, and M15 report method-native
inference latency summaries.

## Deferred methods

- **M15 RLCD-like objective:** deferred by `retrain/15_clef_style_4b/scripts/train_rlcd.sh`.
- **M16 after P0:** teacher, exit distillation, unknown-aware training, risk gate, adaptive serving, LoRA, and VLM are deferred pending P0 evidence. P0 itself uses only inner dev for screening and keeps the 324-row outer test untouched.

M14's V1 training and model-validation path is registered. Its later V2–V5
adaptation stages remain design-only. The remaining deferred methods above are
not included in `all`.
