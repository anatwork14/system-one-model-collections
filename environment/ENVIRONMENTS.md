# Runtime environments

The root `.venv` is the shared data/download/adapter environment. It has Python 3.12 and the dependency snapshot in `requirements.lock`. It is not guaranteed to satisfy every upstream runtime: the pinned SemIf project, for example, pins Torch 2.10, while the current Kev project metadata caps Torch below 2.9. Keep incompatible upstream tools isolated.

For a method runtime, keep the environment outside `upstream/` so upstream source stays read-only. From the workspace root, create a method-local environment and activate it:

```bash
uv venv --python 3.12 released/02_tev1_4b/.venv
source released/02_tev1_4b/.venv/bin/activate
uv sync --project released/02_tev1_4b/upstream --locked --active
```

Use the same pattern with `released/04_kev_4b/upstream` for Kev. Simple Jev's Python project is `upstream/hf-server/`. SemIf has a `pyproject.toml` but no lockfile; select a CUDA-compatible PyTorch wheel first, then install the local package. Nimble's `requirements/training.txt` intentionally omits PyTorch because its published training setup expects a host-provided CUDA wheel. CLM's original package installs vLLM for its serving stack; the controlled projection-head reproduction uses shared Transformers/PyTorch dependencies and does not need vLLM.

Do not replace a project's base CUDA PyTorch with an arbitrary wheel; select a build matching the target host when the workspace is deployed. No target-host GPU checks are required to prepare this workspace.

`requirements.lock` records the root bootstrap as installed in this workspace. It does not pin the method environments and should not be treated as a universal GPU compatibility promise.
