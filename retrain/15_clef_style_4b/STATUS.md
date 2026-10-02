# Clef integration checklist

- [x] Pin official Clef-Flash release and download architecture/head assets.
- [ ] Finish full official Clef-Flash 9B checkpoint download and record hashes.
- [x] Inspect published code/config and generate head tensor shape inventory.
- [x] Initialize a fresh Qwen3.5-4B-dimension head and check mixed-question logit shapes.
- [x] Prepare 2,676 canonical source rows with group-disjoint train/calibration/dev and frozen test.
- [x] Generate train-only question-order variants and validate target membership.
- [ ] Run official 9B functional smoke inference.
- [ ] (Deferred; not needed now) Overfit 32–128 records using the real 4B backbone.
- [ ] (Deferred; not needed now) Train head-only and evaluate the frozen test split.
- [ ] (Deferred; not needed now) Run CE-only and CE+Brier rank-256 LoRA arms.
- [ ] (Deferred; not needed now) Evaluate calibration and multi-question case exact accuracy.
- [ ] (Deferred; optional future work) Consider RLCD-like training after supervised results are stable.

No training is needed for the current request, and none has been run. The current host has an 11 GB GTX 1080 Ti and its installed PyTorch CUDA build is incompatible with the NVIDIA driver, so deferred training would require a supported host. The official 9B model may be smoke-tested on CPU after download.
