# Reproduction protocol — Nimble-4B

This is a Qwen3.5-4B adaptation of Nimble's public Qwen3.5-9B candidate-logit trainer. The 324-row release evaluation split is reserved as an outer holdout. Before a clean training run, `scripts/prepare_inner_split.py` deterministically holds out six complete source families from the original 2,676 training rows (2,268 train / 408 dev), writes a manifest, and builds method-format data for the controlled arms. The trainer receives only this inner train/dev data.

Because candidate token IDs and cached token exports depend on the tokenizer fingerprint, regenerate/verify token exports using the pinned 4B tokenizer before training. The launcher derives optimizer steps from the staged train row count so its step schedule matches the requested one epoch.

Old `checkpoint-335` / `checkpoint-670` VKong runs used the previous data contract and must not be resumed. The legacy outputs are retained as historical artifacts; clean runs use a new output directory.
