#!/usr/bin/env bash
set -euo pipefail
method_dir="$(cd "$(dirname "$0")/.." && pwd)"
checkpoint="${1:-$method_dir/checkpoints/clef4b-head_only-clean}"
output_dir="${2:-$method_dir/results/dev}"
[[ -f "$checkpoint/joint_head.pt" && -f "$checkpoint/run_config.json" ]] || {
  echo "CLEF-style checkpoint is incomplete: $checkpoint" >&2
  exit 2
}
python3 "$method_dir/scripts/predict.py" --checkpoint "$checkpoint" \
  --input "$method_dir/data/dev/nimble.jsonl" --output "$output_dir/predictions.jsonl"
python3 "$method_dir/../../evaluation/metrics/compute_metrics.py" \
  "$output_dir/predictions.jsonl" --output "$output_dir/metrics.json"
python3 "$method_dir/../../evaluation/run_all.py" --validate-jsonl "$output_dir/predictions.jsonl"
