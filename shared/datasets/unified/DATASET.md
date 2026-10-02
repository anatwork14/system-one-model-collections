# Unified decision format

Canonical future format: one JSON object per decision with `id`, `state`, `question`, `options` (ordered list of `{id,text}`), `label`, `domain`, `source_dataset`, `source_split`, and `provenance`. Preserve original labels and include any conversion mapping. Never combine source train/evaluation splits without a recorded split-family check.
