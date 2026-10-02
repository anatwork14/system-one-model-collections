# BOLT HWU64 protocol

This directory contains protocol metadata only; it owns no raw-data copy.

Materialize `known75` first. Define three derived sets:

- **OW-0 Closed:** candidates are known classes only.
- **OW-1 Unknown absent:** known classes plus `UNKNOWN`; target is `UNKNOWN` for held-out classes.
- **OW-2 Runtime supplied:** known classes, the unseen class description, and `UNKNOWN`; target is the supplied unseen class.

The primary split must keep known and unseen class sets disjoint. Record exact BOLT revision, fold, example IDs, and class-description provenance in `splits/`. Never include unseen labels in training class lists.

The pinned BOLT HWU origin rows and `known75` class fold are used directly. The OW views are materialized in `processed/`; the source and resulting split counts are recorded in `processed/protocol_summary.json` and `splits/known75/split_manifest.json`.
