#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${NMR_CRAFT_PYTHON:-python}"
cd "$ROOT"
export PYTHONPATH="$ROOT"
exec "$PYTHON_BIN" -u scripts/run_workflow.py "$@"

