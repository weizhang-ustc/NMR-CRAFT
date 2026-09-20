#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${NMR_CRAFT_PYTHON:-python}"
CONFIG="$ROOT/configs/default.yaml"
HOST="0.0.0.0"
PORT=18080
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--config FILE] [--host HOST] [--port PORT]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"
export PYTHONPATH="$ROOT"
exec "$PYTHON_BIN" -u scripts/faiss_server.py --config "$CONFIG" --host "$HOST" --port "$PORT"
