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

The local host has an 11 GB GTX 1080 Ti and its installed PyTorch CUDA build is incompatible with the NVIDIA driver. A short VKong feasibility run was performed on an A40 on 2026-10-05; details follow.

## 2026-10-05 VKong A40 feasibility run

The rank-256 supervised LoRA launcher passed data validation, downloaded and loaded
the pinned Qwen3.5-4B weights, and executed training on an A40. It reached batch
100 of 6,450 in the approximately 10-minute rental window (training loss 0.5823,
question accuracy 0.7717 at that log point). This is an in-progress training metric,
not a completed evaluation. The run was stopped at the requested rental limit.

No checkpoint was produced or downloaded. `src/train.py` saves its best trainable
state only after completing the full epoch and running inner-dev evaluation; the
transfer service archives artifacts only after a successful process exit. The
10-minute feasibility run therefore cannot yield trained weights. The runner had no
VKong storage, and its temporary `/data` was discarded when the rental stopped.

Validation passed for 2,676 canonical records, 4,300 train-only variants, changed
question order, family-disjoint splits, schema-valid targets, and no test-file read.
The complete training and frozen-test evaluation remain pending. This is a local
reproduction using the pinned 4B backbone and locally chosen LoRA targets, not the
official Clef-Flash 9B model or an official reproduction.

## 2026-10-05 queued rerun

A rerun using the same pinned Qwen3.5-4B rank-256 LoRA setup was queued on an A40,
but the instance remained in `loading` and the Run stayed queued. It was stopped at
the user's request before the GPU session became ready; no new training or artifact
was produced. VKong confirmed the App stopped and active instance count returned to
zero.

A subsequent request tested dev allocations for RTX 3090, RTX 4090, and RTX 5090;
all three returned `the requested development allocation is unavailable` before
quoting or creating compute. An official RTX 3090 Run was then queued at $0.133/hour,
but remained unready with no session and no training logs. It was stopped at the
user's request; VKong reports no active instances. No weights were produced.

## 2026-10-05 RTX 4090 dispatch

An official RTX 4090 run (`app_42b4c25ffc4172e8c17fc8d5`, app `clef4b-m15-lora-4090-20261005`)
was dispatched and selected host `vk_3eef0a8d02ed32f4` at $0.281/hour. The machine
transitioned through `loading` and `offline` while the host provider booted and pulled the
container runtime. At the user's request, the app was stopped; VKong confirmed
termination and active compute returned to 0 instances. No weights were produced.

## 2026-10-05 second RTX 4090 dispatch

A second official RTX 4090 run (`app_ad40bcf18ba263e22a2ec6ef`, `run_492946411f1f9fd0af930067`)
was dispatched and selected host `vk_1ff398fb9451efe1` at $0.307/hour. The machine remained
in `loading` pulling the image without agent registration. At the 25-minute registration
threshold, platform server-side recovery automatically released the host and marked the run
interrupted; active compute returned to 0 instances with full credit preservation ($11.39).
No weights were produced.

## 2026-10-05 third RTX 4090 run (5-minute progress gate)

A third official RTX 4090 run (`app_c26851fcb8a45f92a8787803`, `run_3a020e1f31d485c97a61e06c`)
was launched on host `vk_560a64daa3522aa8` at $0.307/hour following user authentication
verification and data preflight validation. The host was monitored continuously for 5 minutes
from launch (20:22:49 to 20:27:53 local). The machine remained in `loading` (`READY=no`,
`SESSION=no`) without agent registration or container startup. In accordance with the user's
instruction, the run was stopped at the 5-minute mark; VKong confirmed app termination, active
instances returned to 0, and the account balance was recorded at $11.36. No weights were produced.
