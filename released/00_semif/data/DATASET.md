# Data inventory — SemIf / OpenJev

## Source

Recipe: `none`. Source repo/artifact and resolved revision are tracked in `../ARTIFACTS.lock`.

## Local organization

- `raw/`: byte-preserving copies of source files.
- `processed/`: derived examples with transformation script/version.
- `splits/`: frozen row/family IDs and split purpose.
- `manifests/manifest.json`: counts, licenses, label origin, hashes, and model-selection use.

No local data files have been fetched yet unless a file is explicitly listed in the manifest. Never use frozen evaluation records for training or model selection.
