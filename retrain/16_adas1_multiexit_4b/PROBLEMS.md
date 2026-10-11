# M16 problems and training readiness

**Updated:** 2026-10-06  
**Current state:** P0 frozen-layer probe run completed successfully on VKong RTX 4090. Verified report, probes, and manifest are stored in `artifacts/m16-p0-20261006-160007-790492-1913321b1155-results.tar.gz`. The exact archive SHA-256 and operational findings are recorded in [VKONG_RUNBOOK.md](VKONG_RUNBOOK.md).

## Readiness summary

| Area | State | Evidence and consequence |
|---|---|---|
| P0 implementation | Implemented and executed | Fitted frozen-backbone probes at layers 4–32 using CE + Brier; selected checkpoints by inner-dev mean question NLL. |
| P0 data | Prepared and pinned | 2,268 train and 408 dev records; outer 324-row test was not loaded. Labels are synthetic/model-checked and not fully human reviewed. |
| CUDA/runtime | Verified for P0 | Existing RTX 4090; PyTorch 2.8.0+cu128, CUDA 12.8, BF16, Triton CUDA arch 89. The workload reached feature extraction and trained/evaluated all eight probes. |
| Memory | Measured by actual run | GPU total 25,391,333,376 bytes; peak allocated 9,531,049,472 bytes; peak reserved 9,804,185,600 bytes. Do not restore a fixed 24-GiB admission gate. |
| VKong config | Valid for ephemeral P0 | One explicit RTX 4090, no persistent storage, no hourly cap. Config validation does not prove stock; check live provider state and quote before every new rental. |
| P0 artifacts | Downloaded and verified | Archive checksum and all 20 manifest-listed artifacts were verified locally before stopping the app. |
| GPU rental | Released | App `app_2592d46d7268570bc6cfe5c3` stopped; provider confirmed no active Run and zero active instances. |
| Full AdaS1 | Not implemented | Teacher training, multi-exit distillation, gate training, answerability/open-set protocol, staged layer execution, and adaptive serving are not runnable. P0 completion does not mean the full adaptive model is trained. |

## P0 result scope

P0 is a representation screen, not a benchmark claim or evidence of early-exit serving speed. The labels are synthetic/model-checked. Inner-dev results are exploratory and were used to select probe checkpoints; keep the frozen 324-row outer test sealed until exit design and evaluation protocol are frozen. See `METHOD.md`, `RESEARCH_GAP.md`, and the artifact report for details.

## Operational lessons

See [VKONG_RUNBOOK.md](VKONG_RUNBOOK.md) for the verified runtime/config, launch and monitoring procedure, failures and fixes, artifact recovery, and shutdown checks. Key points: CUDA smoke test before dependency work; match the pinned PyTorch/CUDA and extension ABI; install a compiler before Triton preflight when needed; verify remote code version and feature-extraction progress; download and checksum artifacts before stopping; then independently verify zero active instances.

## Next boundary

There is no later M16 training stage registered in `scripts/run_training_methods.py`; it explicitly defers teacher/exits/gate/serve pending P0 evidence. Before another paid GPU run, implement and test the next stage, freeze the exit-selection rule using inner-dev only, specify data and acceptance criteria, and bound duration/spend. Do not start a new rental merely to rerun the completed P0. The previous uncapped-hourly configuration does not define a total cost or runtime ceiling for future stages.
