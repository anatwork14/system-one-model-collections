#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$WORKSPACE_ROOT"
source "$SCRIPT_DIR/vkong_cuda_compat.sh"
sanitize_cuda_compat_path
print_gpu_driver_diagnostics

printf '=== CUDA PREFLIGHT BEGIN ===\n' >&2
python3 - <<'PY'
import os
import traceback
import torch

print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}, flush=True)
try:
    torch.cuda.init()
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    probe = torch.empty((1024,), device=f"cuda:{device}", dtype=torch.float32)
    probe.fill_(1.0)
    torch.cuda.synchronize()
    print({
        "cuda_preflight": "passed",
        "gpu": properties.name,
        "vram_bytes": properties.total_memory,
        "small_allocation": int(probe.numel()),
        "synchronized": True,
    }, flush=True)
except Exception as exc:
    print(f"CUDA preflight failed: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise SystemExit(1)
PY

if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        printf 'Triton needs a C compiler, but apt-get is unavailable.\n' >&2
        exit 1
    fi
    printf '=== INSTALLING TRITON C COMPILER ===\n' >&2
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential
    rm -rf /var/lib/apt/lists/*
fi

python3 -m pip install -r "$SCRIPT_DIR/../requirements-runtime.txt"
CAUSAL_CONV1D_WHEEL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python3 - <<'PY'
import sys
import torch

runtime = {
    "python": sys.version_info[:2],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
}
print({"causal_conv1d_wheel_runtime": runtime}, flush=True)
if runtime != {
    "python": (3, 11),
    "torch": "2.8.0+cu128",
    "cuda": "12.8",
    "cxx11_abi": True,
}:
    raise SystemExit(f"No pinned causal-conv1d wheel matches runtime: {runtime}")
PY
python3 -m pip install --no-deps --no-cache-dir "$CAUSAL_CONV1D_WHEEL_URL"
python3 - <<'PY'
import traceback
import torch
import fla
import causal_conv1d

if not torch.cuda.is_available():
    raise SystemExit("CUDA became unavailable after runtime dependencies were installed")
try:
    import triton
    target = triton.runtime.driver.active.get_current_target()
    print({"triton_target": str(target), "triton_backend": target.backend}, flush=True)
    if target.backend != "cuda":
        raise RuntimeError(f"expected Triton CUDA backend, got {target.backend!r}")
except Exception as exc:
    print(f"Triton CUDA preflight failed: {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise SystemExit(1)
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "flash_linear_attention": getattr(fla, "__version__", "installed"),
    "causal_conv1d": getattr(causal_conv1d, "__version__", "installed"),
}, flush=True)
PY
