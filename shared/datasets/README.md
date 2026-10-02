# Shared dataset registry

Existing model-reproduction datasets and their frozen copies remain under this directory. The broader JEV/open-world/VLM evaluation suite has its own single raw-data owner under [`benchmarks/`](../../benchmarks/README.md), as specified by that registry. Do not create method-local or BOLT-overlay copies of those raw sources.

Every source requires an immutable revision, license/provenance note, observed count, split purpose, SHA-256, label-origin description, and a record of whether it has been used for model selection. Method-local data folders may contain manifests and transformations; do not silently fork dataset contents.
