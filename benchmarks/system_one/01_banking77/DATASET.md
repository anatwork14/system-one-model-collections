# Dataset: BANKING77

## 1. Role in this project
System-One and open-world intent classification.

## 2. Research question
Measure fine-grained 77-way intent classification and candidate routing; the same raw data is referenced by the BOLT overlay.

## 3. Source
Official URL/repository: https://huggingface.co/datasets/PolyAI/banking77
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: 10,003 train; 3,080 test; 13,083 total
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: 77 intents
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: Customer utterance
Question: select or score the correct target for the input.
Criteria/options: Choice(77), label name only and label plus description variants
Expected output: Choice(77), label name only and label plus description variants

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Keep official train and test assignments; never duplicate the raw files for BOLT.

## 11. Metrics
Primary: accuracy, macro-F1, NLL, Brier, ECE
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
`python benchmarks/scripts/download_core.py --dataset banking77` followed by `python benchmarks/system_one/01_banking77/scripts/prepare.py` and `python benchmarks/system_one/01_banking77/scripts/validate.py`.
