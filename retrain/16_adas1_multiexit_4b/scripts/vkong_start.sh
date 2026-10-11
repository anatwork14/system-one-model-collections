#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$WORKSPACE_ROOT"
source "$SCRIPT_DIR/vkong_cuda_compat.sh"
sanitize_cuda_compat_path
print_gpu_driver_diagnostics
export M16_SERVICE_STAGE="${M16_SERVICE_STAGE:-full}"
export M16_EXIT_LAYERS="${M16_EXIT_LAYERS:-12,16,20}"
exec python3 "$SCRIPT_DIR/vkong_m16_service.py"
