# M16 P0 dataset

P0 consumes only the corrected M10 inner split generated from the 2,676 Nimble source training decisions:

- train: 2,268 rows
- inner dev: 408 rows
- frozen outer evaluation: 324 rows; never opened by P0

`scripts/prepare_data.py` maps each original row into M15's typed state/schema format and records the source ID, family, and source file hashes. It checks train/dev separation by row ID, source family, and source ID. It does not synthesize evidence questions or add schema variants. Labels are synthetic/model-checked, not fully human reviewed.

The data source is `retrain/10_nimble_4b/data/inner_split/`. The preparation manifest is written to `data/controlled/manifest.json`. P0 uses dev only for early stopping and comparing layer probes; the outer 324-row evaluation remains reserved for a future frozen final evaluation after the method and exits are fixed.
