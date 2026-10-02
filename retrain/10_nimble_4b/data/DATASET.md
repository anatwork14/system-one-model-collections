# Data inventory — Nimble-4B reproduction

## Source

Recipe: `2,676 train + 324 frozen held-out`. Files are copied byte-for-byte from the pinned `bespokelabsai/nimble` checkout into `raw/`; the canonical shared copy is in `shared/datasets/nimble/`. Source commit and hashes are tracked in `../ARTIFACTS.lock` and `manifests/manifest.json`.

## Local organization

- `raw/`: byte-preserving copies of source files.
- `processed/`: derived examples with transformation script/version.
- `splits/`: frozen row/family IDs and split purpose.
- `manifests/manifest.json`: counts, licenses, label origin, hashes, and model-selection use.

Train and evaluation rows are synthetic and model-checked, not human reviewed. Evaluation families are disjoint from training and must remain frozen; never use these 324 rows for tuning or model selection. Source license and provenance details are in the source manifest and `shared/datasets/nimble/DATASET.md`.
