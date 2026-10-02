# Initial Clef-style surrogate dataset

Run `scripts/prepare_data.py` to derive `clef4b-open-mixture-v1-initial` from the workspace's Nimble 2,676 train and 324 frozen evaluation records. The source train split contains synthetic, model-checked labels. Each canonical record retains its original typed decision and adds binary evidence questions only for certificate atoms marked `supported` or `refuted`.

Source families are assigned as whole groups to train, calibration, and dev (roughly 80/10/10). The separate 324 Nimble eval records remain test only. `scripts/generate_schema_variants.py` reads only the already split training records and permutes question order. The official encoder sorts `choice` IDs lexically, so reordering their input mapping does not change model option order; this reproduction preserves that upstream behavior. Score level order is meaningful and is not shuffled. Validation and test data are never augmented.

The result is a multi-question architecture smoke dataset and a first controlled training source, not Cloudflare's original data and not a human-reviewed production benchmark. Further data sources and programmatic task families are planned as separately versioned additions.
