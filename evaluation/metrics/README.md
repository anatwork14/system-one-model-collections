# Metrics

## Prediction-level metrics

`compute_metrics.py predictions.jsonl` expects one row per decision with a unique `id`, `label`, and normalized `probabilities` map. It checks finite values, target membership, probability sum, and any supplied argmax. It reports accuracy, macro/micro-F1, per-class precision/recall/F1, NLL, multiclass Brier (sum of squared class errors), ECE plus reliability-bin points, MRR/Recall@3/5, AURC, Coverage@1% Risk, and Risk@80/90% Coverage. `--group-by metadata.label` or another row/metadata field reports slices, including per-emotion GoEmotions scores. For CLINC open-set metrics, combine the in-domain `test` and `oos_test` predictions over a common known-plus-`oos` candidate set, then use `--unknown-label oos`; the BOLT unknown class defaults to `UNKNOWN`.

When each input includes both known and unknown targets plus an unknown score (or `UNKNOWN` probability), it also reports unknown AUROC, AUPR-Out, FPR95, known accuracy, unknown precision/recall/F1 at the documented fixed 0.5 threshold, and AUOSCR. `AUOSCR` thresholds the maximum known-class probability and integrates unknown false-positive rate against correctly classified known rate. Unknown AUROC/AUPR/FPR95 use the explicit unknown score. It reports these as unavailable when either class is absent.

## Router quality

`compute_routing_metrics.py router_outcomes.jsonl` expects per-request `selected_model`, candidate outcomes (`success`, optional `utility`), and optional router scores for all candidates. Its input contract is `evaluation/routing_record.schema.json`. It reports selected success, regret (best candidate utility minus selected utility), oracle task-success gap, gain at best (selected routing success minus the strongest fixed model's success), router accuracy, Recall@3/5, and MRR. Regret and oracle gap match when utility is exactly task success; if utility also includes cost or latency, they differ.

## System and multi-objective metrics

`compute_system_metrics.py serving_traces.jsonl` expects one row per request with `method`, `id`, `task_success`, and end-to-end `latency_ms`; see `evaluation/serving_trace.schema.json`. Method comparisons require identical request IDs. Optional measured fields support Cost/request, Cost/success, GPU-sec/request, peak VRAM, QPS/GPU (requires deduplicated run IDs, GPU wall seconds, and GPU count), router overhead ratio, Local Serving Rate, External Offload Rate, and Escalation Rate. Partial optional measurement coverage is reported and leaves the aggregate unavailable. It reports quality retention as each method's mean task quality divided by the best mean quality in the supplied comparison. `--reference-method NAME --quality-floor 0.9` reports CostSave@MatchedQuality only for methods and reference runs that meet the floor. Pareto analysis minimizes cost, p50 latency, and GPU-seconds while maximizing quality. ParetoDist is the Euclidean distance from each method to the closest observed Pareto-front point after per-objective min/max normalization. Distance-to-ideal is reported separately. Normalized utility is reported only when explicit policy weights are supplied with `--utility-weights`.

## Continual learning and robustness

`compute_continual_metrics.py matrix.json` consumes scores for every task before learning and after every training stage using `evaluation/continual_matrix.schema.json`. It reports AA, AIA, average forgetting, BWT, and optionally FWT (requires a supplied baseline score per task) and intransigence (requires a joint-training reference score per task).

`compute_robustness_metrics.py predictions.jsonl` computes in-domain minus OOD accuracy when rows mark `domain_group` as `in_domain` or `ood`. It computes option-permutation consistency only for paired baseline/variant rows with a shared `pair_id`; map displayed option IDs to semantic labels in `choice_semantics`. Without paired runs it reports the metric unavailable.

## Interpretation and limitations

ECE is top-label ECE and depends on the documented bin count (default 15); Brier is the multiclass sum, not divided by number of classes. AURC ranks by maximum class probability and uses empirical risk at each prefix. Coverage@1% Risk is the largest confidence-ranked prefix whose empirical risk is at most 1%; ties are kept in input-stable order. Open-world runtime-supplied unseen accuracy is ordinary target accuracy on OW-2 and must be reported separately from OW-1 unknown detection.

No metric tool creates model outputs, latency traces, cost measurements, continual-learning schedules, or counterfactual permutations. Those values require an adapter/run protocol and must be measured or generated before the corresponding metrics are available. Report option-count, dataset, domain, modality, and protocol slices. Softmax normalization is not calibration.
