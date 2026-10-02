# Dataset: BOLT StackOverflow

## 1. Role in this project
Open-set technical category classification.

## 2. Research question
Test unseen technical categories to reduce dependence on assistant-intent domains.

## 3. Source
Official URL/repository: https://github.com/CNIC-DSL/BOLT
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: BOLT processed projection: 19,985
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: 20 categories
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: StackOverflow text
Question: select or score the correct target for the input.
Criteria/options: Choice(K), UNKNOWN, or runtime-supplied unseen class
Expected output: Choice(K), UNKNOWN, or runtime-supplied unseen class

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Primary known75 fold; retain BOLT class and example split manifests.

## 11. Metrics
Primary: known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy
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
`python benchmarks/scripts/download_core.py --dataset stackoverflow_bolt` followed by `python benchmarks/openworld/13_stackoverflow_bolt/scripts/prepare.py` and `python benchmarks/openworld/13_stackoverflow_bolt/scripts/validate.py`.
