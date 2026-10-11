#!/usr/bin/env bash
# Remove unsupported CUDA forward-compat libraries on consumer RTX rentals.
# Error 804 means the loaded compatibility driver does not support the visible GPU.
sanitize_cuda_compat_path() {
  local original="${LD_LIBRARY_PATH:-}"
  local cleaned
  cleaned="$(python3 -c 'import os,pathlib; parts=os.environ.get("LD_LIBRARY_PATH", "").split(":"); keep=[p for p in parts if p and not any(x == "compat" or x.startswith("cuda-compat") for x in pathlib.PurePosixPath(p).parts)]; print(":".join(keep))')"
  export LD_LIBRARY_PATH="$cleaned"
  if [[ "$original" != "$cleaned" ]]; then
    printf 'Removed CUDA compatibility library path entries unsupported on this GPU.\n' >&2
    printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >&2
  else
    printf 'No CUDA compatibility library path entries found. LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH" >&2
  fi
}

print_gpu_driver_diagnostics() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    nvidia-smi || true
  else
    printf 'nvidia-smi unavailable in container.\n' >&2
  fi
}
