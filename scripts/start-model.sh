#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${NMR_CRAFT_MODEL_PYTHON:-python}"
ROLE=""
MODEL_PATH="$ROOT/ckpt/NMRGym-NMR-LM"
GPU=""
PORT=""
PROMPT_FORMAT="nmrgym"
FORWARD_BATCH_SIZE=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --prompt-format) PROMPT_FORMAT="$2"; shift 2 ;;
    --forward-batch-size) FORWARD_BATCH_SIZE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --role inverse|forward|both --gpu IDS [--model-path PATH] [--port PORT] [--prompt-format FORMAT] [--forward-batch-size N]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROLE" && -n "$GPU" ]] || {
  echo "--role and --gpu are required" >&2; exit 2;
}
if [[ -z "$PORT" ]]; then
  [[ "$ROLE" == "forward" ]] && PORT=18082 || PORT=18081
fi
cd "$ROOT"
export PYTHONPATH="$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" exec "$PYTHON_BIN" -u scripts/model_server.py \
  --role "$ROLE" --model-path "$MODEL_PATH" --port "$PORT" \
  --prompt-format "$PROMPT_FORMAT" --forward-batch-size "$FORWARD_BATCH_SIZE"
