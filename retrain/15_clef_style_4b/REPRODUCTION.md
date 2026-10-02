# Clef-style 4B reproduction status

## Completed preparation

1. Pin and download only Cloudflare's Clef-Flash release.
2. Inspect the published head implementation, config, and tensor shapes.
3. Instantiate a fresh head using the local Qwen3.5-4B hidden width and verify mixed question logits.
4. Convert the Nimble controlled set to canonical typed records; make family-disjoint train/calibration/dev splits and preserve its frozen evaluation set as test.
5. Generate question and choice-option ordering variants from training records only.

## Future training sequence (deferred; not needed now)

1. Run `scripts/smoke_test.py` for the cheap head shape check.
2. On supported training hardware, overfit a manually selected 32–128 train-record subset with `src/train.py --stage head_only --max-records 128`.
3. Train the head-only baseline on the train split.
4. Compare CE-only and CE+Brier; set `--brier-weight 0` for CE-only.
5. Enable `--stage lora --rank 256` only after the head-only checks succeed.
6. Evaluate calibration on `data/calibration/`; keep `data/test/nimble_frozen.jsonl` untouched until final evaluation.
7. Defer any RLCD-like objective until supervised training and calibration are stable. Label it as a local reproduction.
8. Add the 40K+ multi-source mixture and multimodal training only in later, separately versioned stages.

The local LoRA targets, optimizer settings, Brier coefficient, and augmentation recipe are experimental choices. They are not Cloudflare's unpublished settings. Current workspace host is not able to run the prescribed 4B GPU training; no training result is claimed.
