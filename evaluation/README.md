# Evaluation protocol

For the exact training commands, CUDA requirements, report layout, and method
coverage, see the [training runbook](../TRAINING_RUNBOOK.md).

`run_all.py --list` inventories documented methods and marks the deferred VLM arm. `scripts/run_training_methods.py` runs the five supported local training methods and their native inner-dev evaluations. `standardize_training_eval.py` adapts M10–M13 probability outputs for shared classification/calibration/selective metrics. Those metric inputs are intentionally not full serving records: they do not invent model revisions, token counts, or serving measurements. A complete serving adapter must emit the [normalized prediction record](result_record.schema.json). The probability map must be finite, include the target, and sum to one. Missing system measurements must remain missing, not be written as zero.

## Training method runner

Run one local method and its native inner-dev evaluator with:

```bash
python3 scripts/run_training_methods.py M10
```

Use `M10`, `M11`, `M12`, `M13`, or supervised `M15`. Pass `all` to run those
methods serially. Each invocation writes logs, copied native artifacts, and
`standard_report.json` under `results/training_runs/<run-id>/`. Training runs
by default; `--skip-train` evaluates an existing expected checkpoint. The
wrapper sets each train script's required `CONFIRM_TRAIN=yes` internally. It
does not invoke deferred M14 or the deferred M15 RLCD-like script.

M10–M15 report the shared decision metrics on each method's own family-disjoint
inner-dev split; the splits differ and are not a cross-method benchmark
comparison. M15 inference reuses its saved joint head and, for the LoRA stage,
the saved adapter. Routing, open-set, open-world,
system, cost, compute, local-serving, multi-objective, and robustness metrics
are marked unavailable unless their required traces are collected. M13's
native latency summary is preserved; missing p99 and GPU measurements are not
inferred.

The released smoke case in `datasets/released_smoke.jsonl` is infrastructure validation only. It is not a benchmark and must not be used to tune prompts or thresholds. The metric tools under `metrics/` report classification, macro-F1, ranking, calibration, selective-risk, open-set, routing, continual-learning, robustness, and system metrics when their required traces exist. A metric omitted from an output means its input fields were unavailable; do not impute a value.

Never compare `UPSTREAM_*` and `OURS_*` results without displaying that identity in the table. Record raw per-example outputs and a dataset manifest/hash alongside aggregates. The result directory currently has no benchmark inference runs, so these tools define evaluation and do not imply model results exist.
