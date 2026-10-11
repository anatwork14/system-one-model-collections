# Survey questions and benchmark coverage

## What and which decisions are being evaluated

The normalized suite defines decision tasks with an input state, a question, candidate criteria, and a target. It spans native structured JEV decisions; intent and semantic classification; Boolean decisions; 28 independent GoEmotions decisions per text; direct/tool/request-info/cannot-answer routing; tool choice with a `NO_TOOL` option; open-set class discovery; and image classification at 10, 37, and 102 classes. The exact per-source mapping and limitations are in each dataset card.

This answers what tasks the current suite can measure and which candidate decision wins for each task. It does not establish one universally best training method: no shared benchmark inference outputs exist yet, exposure to benchmark training data is not registered for each method, and the methods have not all completed controlled retraining. Training recipes must be compared on identical examples, splits, revisions, and evaluation conditions before selecting one.

## Metric coverage and required evidence

| Survey axis | Implemented metric tool | Data/protocol available | Current evidence status |
|---|---|---|---|
| Routing quality | `evaluation/metrics/compute_routing_metrics.py` | When2Call four-way decisions, BFCL tool choice/NO_TOOL, candidate outcomes for each model | Dataset records exist; no router/model outcome traces yet |
| Decision calibration | `compute_metrics.py` | Labeled prediction probabilities | Tool supports NLL, Brier, ECE, reliability-bin values; no model predictions yet |
| Open-set | `compute_metrics.py` | Native CLINC OOS plus BOLT OW-1 unknown-absent and OW-2 runtime-supplied partitions | Dataset protocols exist; no predictions yet. Report OW-1 detection separately from OW-2 supplied-unseen accuracy |
| Selective decision | `compute_metrics.py` | Confidence-ranked labeled predictions | Tool supports AURC, Coverage@1% Risk, and Risk@80/90% Coverage; no predictions yet |
| Open-world learning | `compute_continual_metrics.py` | Requires a fixed task order, full accuracy matrix, initialization baselines, and joint-training references | Not currently represented by a continual training/evaluation protocol |
| System quality | `compute_system_metrics.py` | Per-request task success or quality trace | No served-request traces yet; JEV/BFCL need task-specific evaluators for full success/quality |
| Cost | `compute_system_metrics.py` | Per-request measured USD, and a reference method/quality floor for matched-quality saving | No cost traces or cost model yet |
| Latency | `compute_system_metrics.py` | Per-request end-to-end timing and router timing | No latency traces yet; tool reports p50/p95/p99 and router overhead when measured |
| Compute | `compute_system_metrics.py` | GPU seconds/request, per-run GPU wall time, peak VRAM | No hardware traces yet; QPS/GPU requires run-level wall time and GPU identity in run metadata |
| Local AI | `compute_system_metrics.py` | Boolean local serving, external offload, and escalation outcomes for every request | No deployment traces yet |
| Multi-objective | `compute_system_metrics.py` | Comparable per-method quality, cost, latency, GPU-seconds | Pareto front, ParetoDist, optional weighted normalized utility; no comparable measurements yet |
| Robustness | `compute_robustness_metrics.py` | Explicit in-domain/OOD groups and paired baseline/permuted-option predictions | OOD sources and train-only schema permutations exist, but test-time paired perturbation predictions are not generated |

## Data and leakage checks

The dataset registry currently lists 14 benchmark entries over 12 unique raw sources: seven System-One sources, four BOLT protocols, and three vision sources. The BOLT overlays reuse canonical raw data where possible and split known/unseen classes with the pinned `known75` manifests. Native CLINC OOS and BOLT-held-out CLINC intents are distinct protocols. Training/evaluation exposure remains unknown until model-run registrations record each method's source data.

The class and image datasets support broad model selection, but some benchmark rows are also training data for public/released models. Do not interpret scores as unseen generalization until each model's exposure is declared. Use validation splits for model selection; reserve test/OOS test for final reporting. Keep OW-2 runtime-supplied examples out of class descriptions seen during training.

## Evidence boundary

All benchmark datasets have normalized records and preparation/validation manifests. Those are data-integrity results, not model results. `results/released/released_smoke.csv` contains only its header; no common adapter has emitted prediction traces; no router replay, serving-cost experiment, continual-learning schedule, or final benchmark table is complete. Metrics are computable only after the corresponding evidence is recorded under the result-record contract in `evaluation/result_record.schema.json` and the specialized trace formats documented in `evaluation/metrics/README.md`.
