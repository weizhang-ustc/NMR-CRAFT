#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INVERSE_GPU=0
FORWARD_GPU=1
MODEL_PATH="$ROOT/ckpt/NMRGym-NMR-LM"
FORWARD_BATCH_SIZE=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inverse-gpu) INVERSE_GPU="$2"; shift 2 ;;
    --forward-gpu) FORWARD_GPU="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --forward-batch-size) FORWARD_BATCH_SIZE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--inverse-gpu ID] [--forward-gpu ID] [--model-path PATH] [--forward-batch-size N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$ROOT/logs"
"$ROOT/scripts/start-model.sh" --role inverse --gpu "$INVERSE_GPU" \
  --model-path "$MODEL_PATH" >"$ROOT/logs/inverse-model.log" 2>&1 &
INVERSE_PID=$!
"$ROOT/scripts/start-model.sh" --role forward --gpu "$FORWARD_GPU" \
  --model-path "$MODEL_PATH" --forward-batch-size "$FORWARD_BATCH_SIZE" \
  >"$ROOT/logs/forward-model.log" 2>&1 &
FORWARD_PID=$!

cleanup() {
  kill "$INVERSE_PID" "$FORWARD_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "inverse pid=$INVERSE_PID gpu=$INVERSE_GPU port=18081"
echo "forward pid=$FORWARD_PID gpu=$FORWARD_GPU port=18082 batch=$FORWARD_BATCH_SIZE"
wait -n "$INVERSE_PID" "$FORWARD_PID"
