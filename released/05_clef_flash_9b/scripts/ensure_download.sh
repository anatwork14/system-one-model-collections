#!/usr/bin/env bash
set -euo pipefail

# Restart the resumable snapshot downloader if the current process exits.
watched_pid="${1:?usage: ensure_download.sh CURRENT_DOWNLOAD_PID}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$script_dir/../../.." && pwd)"

while kill -0 "$watched_pid" 2>/dev/null; do
  state="$(ps -o stat= -p "$watched_pid" 2>/dev/null | awk '{print $1}')"
  command="$(ps -o args= -p "$watched_pid" 2>/dev/null || true)"
  [[ "$state" == Z* ]] && break
  [[ "$command" == *"released/05_clef_flash_9b/scripts/download_full.py"* ]] || break
  sleep 20
done

cd "$workspace"
export HF_HUB_DISABLE_XET=1
exec "$workspace/.venv/bin/python" "$script_dir/download_full.py"
