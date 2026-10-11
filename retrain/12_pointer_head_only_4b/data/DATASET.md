# Data inventory — Pointer-head-only-4B

## Source

Controlled Nimble-derived decisions. The frozen 2,676-record training split and
324-record evaluation split are copied byte-for-byte from
`retrain/10_nimble_4b/data/processed/kev_{train,eval}.jsonl`; training labels come
from the prepared Nimble references. No evaluation examples are included in training.

## Local organization

- `raw/`: byte-preserving copies of source files.
- `processed/`: derived examples with transformation script/version.
- `splits/`: frozen row/family IDs and split purpose.
- `manifests/manifest.json`: counts, licenses, label origin, hashes, and model-selection use.

The split sizes and SHA-256 checksums are recorded in `manifests/manifest.json`.
Never use frozen evaluation records for training or model selection.
