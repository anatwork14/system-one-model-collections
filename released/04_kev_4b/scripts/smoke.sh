#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PORT="${PORT:-18004}"
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
PYTHONPATH="$HERE/upstream${PYTHONPATH:+:$PYTHONPATH}" \
  python -m kev.serve --run "$HERE/weights/released" --host 127.0.0.1 --port "$PORT" \
  >"$HERE/results/server.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/v1/systemone" -H 'Content-Type: application/json' \
  --data-binary @"$HERE/../../evaluation/datasets/kev_smoke.json" | tee "$HERE/results/smoke.json"
