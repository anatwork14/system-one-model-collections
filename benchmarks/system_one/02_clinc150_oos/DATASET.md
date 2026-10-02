# Dataset: CLINC150 + OOS

## 1. Role in this project
System-One classification and native out-of-scope detection.

## 2. Research question
Measure 150 intent classes, native OOS detection, abstention, and high-cardinality choices. Native OOS and BOLT-held-out intents are distinct protocols.

## 3. Source
Official URL/repository: https://github.com/clinc/oos-eval
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: 23,700 labeled records: 150 in-scope classes plus OOS
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: 150 intents plus OOS
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: User utterance
Question: select or score the correct target for the input.
Criteria/options: Choice(150); OOS is a separate unknown target
Expected output: Choice(150); OOS is a separate unknown target

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Use data_full.json only; preserve official train, validation, and test splits.

## 11. Metrics
Primary: accuracy, macro-F1, OOS AUROC/AUPR, NLL, Brier, ECE
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
`python benchmarks/scripts/download_core.py --dataset clinc150_oos` followed by `python benchmarks/system_one/02_clinc150_oos/scripts/prepare.py` and `python benchmarks/system_one/02_clinc150_oos/scripts/validate.py`.
