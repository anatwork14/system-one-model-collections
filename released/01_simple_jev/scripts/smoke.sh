#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PORT="${PORT:-18001}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LOG="$HERE/results/server.log"
python "$HERE/upstream/hf-server/hf_server.py" \
  --model "$ROOT/shared/models/qwen3.5-4b" --device "$DEVICE" --dtype "$DTYPE" \
  --classifier-prompt-policy baseline --max-model-len 2048 --port "$PORT" >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/v1/classifier" -H 'Content-Type: application/json' \
  --data-binary @"$HERE/../../evaluation/datasets/simple_jev_smoke.json" | tee "$HERE/results/smoke.json"
