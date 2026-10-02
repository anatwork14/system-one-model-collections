# Reproduction protocol — Nimble-4B

This is a Qwen3.5-4B adaptation of Nimble's public Qwen3.5-9B candidate-logit trainer. Keep the 324-row evaluation set frozen. Run preparation first, inspect `data/manifests/manifest.json`, then perform a 32–64 example overfit sanity run before the full epoch. That sanity run is future work and is not part of workspace preparation.

The only planned method change is backbone size. Because candidate token IDs and any cached token exports depend on tokenizer fingerprint, regenerate/verify token exports using the pinned 4B tokenizer before training.
