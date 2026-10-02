# Dataset: JevBench Public

## 1. Role in this project
System-One native decisions.

## 2. Research question
Evaluate Choice, Score, Noul, routing, policy, numeric, probability, ambiguity, and multi-hop behavior. Only the 231 public examples are in scope; 303 tasks are private or held out.

## 3. Source
Official URL/repository: https://github.com/fstandhartinger/jevbench
Pinned commit/revision: see the immutable revision in `SOURCE.lock.yaml`
Downloaded date: 2026-10-02

## 4. Public availability
Public: yes; access and use remain subject to upstream terms.
License/terms: verify and record upstream license before redistribution.
Redistribution allowed: not assumed; raw artifacts remain local and are not redistributed by this repository.

## 5. Dataset size
Expected: 231 public (72 original, 48 easy, 111 hard)
Observed: downloaded; see `manifests/download_report.json` for split counts and checksum validation

## 6. Classes / labels
Expected: Variable native outputs
Class names location: source-defined labels and `processed/` manifest after preparation.

## 7. Original task
See the upstream source and its documentation.

## 8. Our System-One mapping
Input state: Native decision record
Question: select or score the correct target for the input.
Criteria/options: evaluation only; no training
Expected output: evaluation only; no training

## 9. Example
Canonical normalized records are available under `processed/` after preparation; records retain source IDs and original split names.

## 10. Benchmark protocol
Pin the repository commit, then copy only public task files into raw/.

## 11. Metrics
Primary: source-specific native evaluator
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
`python benchmarks/scripts/download_core.py --dataset jevbench_public` followed by `python benchmarks/system_one/00_jevbench/scripts/prepare.py` and `python benchmarks/system_one/00_jevbench/scripts/validate.py`.
