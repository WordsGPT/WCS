#!/usr/bin/env bash
set -euo pipefail

# One-command unquantized Gemma 3 27B WCS run.
# Override any setting inline, for example:
#   LIMIT=5 ./goCarlos
#   MODELS=gemma3-27b-base,gemma3-27b-it ./goCarlos

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-gemma3-27b}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

SAMPLES="${SAMPLES:-data/processed/samples.jsonl}"
MODELS="${MODELS:-gemma3-27b-base}"
TEMPERATURES="${TEMPERATURES:-1.0,0.7,1.5}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_MEMORY="${MAX_MEMORY:-0=44GiB,1=44GiB,cpu=160GiB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-offload/gemma3-27b}"
RESULTS_DIR="${RESULTS_DIR:-results/gemma3_27b_unquantized}"
LOGS_DIR="${LOGS_DIR:-logs/gemma3_27b_unquantized}"
RETRIES="${RETRIES:-1}"
LIMIT="${LIMIT:-}"

mkdir -p "$LOGS_DIR" "$OFFLOAD_FOLDER"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$INSTALL_DEPS" = "1" ]; then
  python -m pip install --upgrade pip
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
  python -m pip install --upgrade "transformers>=4.51.0" accelerate sentencepiece protobuf safetensors
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. If Gemma access is gated on this machine, run:"
  echo "  huggingface-cli login"
  echo "or rerun with HF_TOKEN=..."
fi

COMMAND=(
  python -u scripts/run_model_suite.py
  --samples "$SAMPLES"
  --models "$MODELS"
  --temperatures "$TEMPERATURES"
  --results-dir "$RESULTS_DIR"
  --logs-dir "$LOGS_DIR"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --device-map "$DEVICE_MAP"
  --max-memory "$MAX_MEMORY"
  --offload-folder "$OFFLOAD_FOLDER"
  --retries "$RETRIES"
  --trust-remote-code
)

if [ -n "$LIMIT" ]; then
  COMMAND+=(--limit "$LIMIT")
fi

echo "Running unquantized Gemma 3 27B WCS audit"
echo "Models: $MODELS"
echo "Dtype: $DTYPE; device_map: $DEVICE_MAP; max_memory: $MAX_MEMORY"
echo "Results: $RESULTS_DIR"
echo "Logs: $LOGS_DIR"
echo

"${COMMAND[@]}" 2>&1 | tee "$LOGS_DIR/run.log"
