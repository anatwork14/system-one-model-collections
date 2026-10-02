# Tev1 new-v1 dataset

The pinned upstream build recipe reports 37,840 training and 4,568 development examples. Core sources include MultiNLI, BoolQ, Banking77, AG News, SST-5, synthetic policy/routing decisions, and synthetic research classification. Each public record includes its source repository, revision, original split, and source row index; synthetic records carry their executable rule/generator provenance.

The upstream source notes mixed/unknown/unspecified licenses for some source corpora (notably AG News and SST-5). Do not redistribute the combined dataset or claim commercial clearance based only on the Tev1 code license. Dataset labels include source labels and generated construction labels; these are not uniformly human-reviewed decisions. Exact source dataset revisions and license metadata are in `released/02_tev1_4b/data/manifests/tev1_sources.lock.json`.

Rebuild into method-local staging/output under `released/02_tev1_4b/data/`, never into the upstream checkout. Validate generated manifests and hashes before any use.
