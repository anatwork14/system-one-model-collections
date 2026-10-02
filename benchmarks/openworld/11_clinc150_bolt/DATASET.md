# Dataset: BOLT CLINC150 protocol

## 1. Role in this project
Open-set intent classification overlay.

## 2. Research question
Apply BOLT class holdouts to CLINC intents; this is distinct from native CLINC OOS examples.

## 3. Source
Official URL/repository: https://github.com/CNIC-DSL/BOLT
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: BOLT processed projection: 22,495
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: 150 intents
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: Reference to CLINC utterances
Question: select or score the correct target for the input.
Criteria/options: Closed known classes; UNKNOWN; or runtime-supplied unseen class
Expected output: Closed known classes; UNKNOWN; or runtime-supplied unseen class

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Primary known75 fold; raw_ref points to system_one/02_clinc150_oos/raw.

## 11. Metrics
Primary: known accuracy, unknown AUROC/AUPR, FPR@95TPR, OSCR, supplied-unseen accuracy
Secondary: report per-class results and calibration metrics when target probabilities are available.

## 12. Preprocessing
Source-specific transformations are versioned in `scripts/prepare.py`; raw source files are immutable.

## 13. Leakage rules
Evaluation splits must not enter model training or model selection. Any intentional in-domain training must be called out in the experiment manifest. Unseen class labels must never appear in training class lists.

## 14. Files
`raw/` stores the one canonical raw copy when this dataset owns raw data. `processed/` stores normalized records. `splits/` stores frozen IDs and protocol manifests.

## 15. Checksums
`checksums.sha256` is generated after download; raw payload SHA-256 entries are recorded in `SOURCE.lock.yaml` and `checksums.sha256`.

## 16. Known limitations
Counts were checked against the pinned source revision; see `manifests/download_report.json`. Label ambiguity, domain bias, and source-specific limitations must be recorded after inspection.

## 17. Reproduction command
`python benchmarks/scripts/download_core.py --dataset clinc150_bolt` followed by `python benchmarks/openworld/11_clinc150_bolt/scripts/prepare.py` and `python benchmarks/openworld/11_clinc150_bolt/scripts/validate.py`.
