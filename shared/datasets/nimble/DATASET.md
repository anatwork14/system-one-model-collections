# Nimble original-2676 controlled dataset

Source: `bespokelabsai/nimble`, pinned `main` commit recorded in `retrain/10_nimble_4b/ARTIFACTS.lock`, files `data/train.jsonl`, `data/eval.jsonl`, and `data/manifest.json`.

Expected counts: 2,676 training and 324 frozen evaluation records. Ten subject domains and Choice, Noul/Boolean, and Score task types. The source describes labels as synthetic and model-checked; people have not reviewed every label. See `manifest.json` for local file hashes and the source manifest for license/provenance details.

The evaluation split is frozen final evaluation only. Do not train on it or use it for prompt/hyperparameter selection.
