# Dataset: GoEmotions

## 1. Role in this project
System-One multi-label and fan-out decisions.

## 2. Research question
Evaluate 28 independent emotion decisions per text, including Neutral, and compare independent scoring with joint schema heads.

## 3. Source
Official URL/repository: https://github.com/google-research/google-research/tree/master/goemotions; raw CSVs are from the linked Google Cloud Storage files
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Observed: 58,011 distinct source IDs across 211,225 annotation rows; filtered train/dev/test: 43,410/5,426/5,427. The planning document listed 58,009 raw examples, two fewer than the pinned source. The source annotation rows, Hub copy, and filtered splits agree.

## 6. Classes / labels
Expected: 27 emotions plus Neutral
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: Reddit comment
Question: select or score the correct target for the input.
Criteria/options: 28 Noul questions per example
Expected output: 28 Noul questions per example

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Use the filtered split; retain multi-label targets and document label mapping.

## 11. Metrics
Primary: micro/macro-F1, per-label AUROC, NLL/Brier/ECE where defined
Secondary: report per-class results and calibration metrics when target probabilities are available.

## 12. Preprocessing
Source-specific transformations are versioned in `scripts/prepare.py`; raw source files are immutable.

## 13. Leakage rules
Evaluation splits must not enter model training or model selection. Any intentional in-domain training must be called out in the experiment manifest. 

## 14. Files
`raw/` stores the one canonical raw copy when this dataset owns raw data. `processed/` stores normalized records. `splits/` stores frozen IDs and protocol manifests.

## 15. Checksums
`checksums.sha256` is generated after download; raw payload SHA-256 entries are recorded in `SOURCE.lock.yaml` and `checksums.sha256`.

## 16. Known limitations
Counts were checked against the pinned source revision; see `manifests/download_report.json`. Label ambiguity, domain bias, and source-specific limitations must be recorded after inspection.

## 17. Reproduction command
`python benchmarks/scripts/download_core.py --dataset goemotions` followed by `python benchmarks/system_one/04_goemotions/scripts/prepare.py` and `python benchmarks/system_one/04_goemotions/scripts/validate.py`.
