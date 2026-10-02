# Dataset: Oxford Flowers-102

## 1. Role in this project
VLM high-cardinality fine-grained image classification.

## 2. Research question
Measure semantic visual classification at 102-way cardinality.

## 3. Source
Official URL/repository: https://huggingface.co/datasets/nelorth/oxford-flowers
Pinned commit/revision: see `SOURCE.lock.yaml`; payload transferred from the pinned Hugging Face mirror `Donghyun99/Oxford-Flower-102` because the official Oxford host was too slow to complete reliably.
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: 8,189 images with official train/validation/test split
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: 102 classes
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: Image
Question: select or score the correct target for the input.
Criteria/options: Choice(102) using class names/descriptions
Expected output: Choice(102) using class names/descriptions

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Preserve official split; test images are evaluation-only.

## 11. Metrics
Primary: top-1 accuracy, macro-F1, NLL, Brier, ECE
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
`python benchmarks/scripts/download_core.py --dataset flowers102` followed by `python benchmarks/vision/22_flowers102/scripts/prepare.py` and `python benchmarks/vision/22_flowers102/scripts/validate.py`.
