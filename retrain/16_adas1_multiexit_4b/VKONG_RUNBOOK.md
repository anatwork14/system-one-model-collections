# M16 VKong Runbook — lessons from P0

## Known-good launch shape

- Use the existing root `vkong.yaml` source root (`retrain/16_adas1_multiexit_4b`) and one explicit GPU: `RTX 4090`. Keep `storage` absent for an ephemeral run; this P0 recovered its result through the service artifact endpoint, so no persistent volume was needed.
- The verified runtime combination was `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`, Python 3.11, PyTorch `2.8.0+cu128`, CUDA 12.8, BF16. The RTX 4090 exposed about 25.39 GB; peak allocated/reserved memory was about 9.53/9.80 GB for this probe. Don't use a nominal 24-GiB admission gate; log free/total memory and test the real workload.
- In setup, first sanitize CUDA compatibility-library paths and run `torch.cuda.init()`, a small allocation, and `torch.cuda.synchronize()`. Install a C compiler before dependency setup when absent, then install pinned runtime requirements and the Torch/CUDA/Python/ABI-matched prebuilt `causal-conv1d` wheel. Explicitly require Triton's active backend to be CUDA before starting the worker.
- Reuse a confirmed live rental when one exists. Validate actual `vkong app show`, `vkong instances list`, and `vkong whoami`; do not infer GPU availability from local YAML validation or a stale offer. For a redeploy on a retained instance, use `vkong attach <instance> -C . --update`; verify the provider Run and live service status/logs after launch.
- Stream child output to both provider logs and the durable service log. Verify `/status` advances to `training`, inspect `/logs` for extracted records and epoch metrics, then wait for `ready` and the archive checksum.
- Download `/artifact` before stopping the app. Compare the archive SHA-256 to `/status`, verify every manifest artifact's byte count and SHA-256, inspect the tar members, and only then stop the exact app and confirm zero active instances.

## Failures and fixes observed

- CUDA error 804 / forward-compatibility failure: the first image/runtime pairing was incompatible. Matching the known M15 PyTorch CUDA 12.8 runtime and adding a CUDA init/allocation/synchronize preflight let the target RTX 4090 pass.
- Missing `joint_schema_model` import: the helper was not inside the uploaded source tree. Stage it under M16's uploaded `vendor/architecture/` directory and verify the runtime tree/import.
- Causal-conv1d source build and runtime mismatch risk: replace compilation with the pinned prebuilt wheel and assert Python, Torch, CUDA, and C++ ABI before install.
- Triton fell back / failed because it could not find a compiler or CUDA backend: install `build-essential` before pip dependencies when needed, use the FLA CUDA extra, and fail setup unless Triton reports CUDA (`arch=89` on the 4090).
- RTX 4090 reported ~23.6 GiB in the old check and was rejected by a strict 24-GiB gate despite the actual probe fitting. Remove the fixed threshold; the successful probe later peaked below 10 GB allocated.
- Stale RTX 4090 offers became unavailable: this was placement/supply failure, not evidence of a source-code defect. Check the currently rented instance first; don't create parallel rentals or switch GPU classes.
- Old P0 code collided with an existing output directory on the retained VM (`FileExistsError`). Service and worker both tried to create it. Remove the duplicate service-side creation and allocate collision-resistant run IDs; regression-test reuse of a pre-existing run directory.
- The first corrected run still reached old uploaded code. A later `--update` deploy produced a fresh run ID; verify the remote log includes the expected code behavior, not just local edits or app startup.
- Local tunnel `/status`/`/logs` can time out or reset when a worker exits; check provider app/run state and reconnect rather than treating a broken tunnel as proof the run failed. In this successful run, the service remained live long enough to recover the artifact.
- `curl -o` failed when the local `artifacts/` directory did not exist. Create the destination directory before downloading.
- A ready archive does not mean the GPU rental has stopped. Stop only after download/checksum validation, then verify `active: 0` and `active_run: null`.

## Verified P0 outcome

- VKong app `app_2592d46d7268570bc6cfe5c3`; provider Run `run_63d6288df914a4036ace9a65`; instance `vk_19a1f3feb1e950d3`; RTX 4090; observed rate `$0.404149/hour`.
- P0 extracted the 2,268 train and 408 dev records and trained probes at layers 4, 8, 12, 16, 20, 24, 28, and 32. It did not load the frozen 324-row outer test.
- Archive: `artifacts/m16-p0-20261006-160007-790492-1913321b1155-results.tar.gz`; SHA-256 `6a0673268a485c6ba4abbf6a1d5a18132b13cce63a18d58bec283459d7e99305`. All 20 manifest-listed artifacts were verified against their recorded sizes and hashes.
- The exact app was stopped after verification; VKong reported zero active instances.
- The P0 labels are synthetic/model-checked, not fully human reviewed. This is a representation screen, not evidence of an adaptive serving speedup or final benchmark.

## Next training boundary

P0 is complete. The project currently registers only `--stage probe`; teacher training, multi-exit distillation, gate training, staged execution, and adaptive serving are not implemented. Do not rent another GPU expecting those later stages to run until their code, data/protocol, acceptance criteria, and bounded execution plan are implemented and validated. Keep the outer 324-row test sealed while selecting exits and designing those stages.
