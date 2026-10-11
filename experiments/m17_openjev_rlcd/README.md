# M17: adapted OpenJev-RLCD on Qwen3.5-4B

Implementation status: source pinned; adaptation and verification in progress. No
training has run yet. See `PATCH_NOTES.md` and the pinned config for intended semantics.

The experiment uses the original GSM8K-Verify file from `upstream/data/` without
rebuilding it. Training/dev selection follows upstream `load_items(split_seed=0,
n_train=4000, n_dev=500)`; the original GSM8K test split is reserved for final evaluation.

## Planned entry points

- `scripts/run_m17_qwen35_4b.sh`: single Stage-1/Stage-2 pair.
- `scripts/vkong_transfer_service.py`: VKong service wrapper that exposes logs and
  completed artifacts over the attached tunnel so outputs can be exported without
  enabling experimental VKong Storage.
- `scripts/train_m17.py`: Qwen3.5/LoRA training and evaluation adaptation.

GPU provisioning requires a live VKong quote and explicit approval for that exact
configuration. No allocation is authorized by a general training request.
