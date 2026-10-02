# Data inventory — Tev1-4B

## Source

Recipe: `new-v1`. Source repo/artifact and resolved revision are tracked in `../ARTIFACTS.lock`.

## Local organization

- `raw/`: byte-preserving copies of source files.
- `processed/`: derived examples with transformation script/version.
- `splits/`: frozen row/family IDs and split purpose.
- `manifests/manifest.json`: counts, licenses, label origin, hashes, and model-selection use.

The exact upstream commit is archived under `data/build_src/`; the build output is copied into `original_recipe/new-v1/`. The build manifest records public source revisions, output split counts, and content hashes. This locally materialized recipe is distinct from the published checkpoint provenance.
