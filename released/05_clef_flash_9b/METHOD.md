# Cloudflare Clef-Flash 9B

## Identity

This folder is the official Cloudflare reference release `Cloudflare/clef-flash` at Hub revision `17f0b0ad64efb65d273590632833508766b2aae6`. It is the 9B Qwen3.5 model. It is not a 4B checkpoint and is not a locally trained artifact.

## Published method

Clef-Flash takes one shared state and one or more typed questions. The state can be text or JSON and may include images or video. Supported question types are `choice`, `score`, and `noul` (yes/no). Each question gets logits only for its schema options; softmax is applied independently per question. It does not decode a generated answer.

The released `JointSchemaHead` consumes final backbone hidden states. Option representations route to evidence in the prompt. Per-question field vectors then interact through a transformer decoder and attend back to the state before a schema-bound scorer returns option logits. The implementation includes an output-embedding lexical prior. The head has 2 evidence-routing layers, 4 field decoder layers, width 1,024, 16 attention heads, and feedforward width 4,096. The published head config has `hidden_size=4096` for the 9B backbone.

Cloudflare reports a frozen Qwen3.5-9B backbone with jointly optimized rank-256 low-rank adapters and the schema head. It reports label-smoothed cross-entropy plus Brier calibration loss, followed by RLCD. Training data is described as internal synthetic data with field, prompt, and schema permutations. Public details do not specify the source records, split, loss coefficients, adapter target modules, or exact RLCD objective.

## Local artifact arrangement

- `weights/architecture/` contains the small upstream code, head, config, and tokenizer metadata needed for inspection.
- `weights/head/` exposes the released head weights and config through links to the same files in `weights/architecture/`.
- `weights/model/` contains the complete pinned Cloudflare Clef-Flash release, including its merged Qwen3.5-9B weights.
- `audit/` records architecture and tensor inspection. A Qwen-vs-Clef parameter diff requires a separate local Qwen3.5-9B directory; this workspace intentionally does not fetch that second 9B checkpoint.

The license is Apache-2.0. The original training data and full training recipe are not public, so exact training reproduction is not possible from this release.

## Reproduction command

From the workspace root:

```bash
.venv/bin/python released/05_clef_flash_9b/scripts/download_minimal.py
.venv/bin/python released/05_clef_flash_9b/scripts/download_full.py
```

The scripts pin the same Hub revision. The full local checkpoint is the official model; do not label the separate 4B arm as “Cloudflare Clef-4B.”
