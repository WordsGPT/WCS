#!/usr/bin/env bash
set -euo pipefail

# One-command unquantized Gemma 3 27B WCS run.
# Override any setting inline, for example:
#   LIMIT=5 ./goCarlos
#   MODELS=gemma3-27b-base,gemma3-27b-it ./goCarlos

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv-gemma3-27b}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
MIN_PYTHON="${MIN_PYTHON:-3.11}"

SAMPLES="${SAMPLES:-data/processed/samples.jsonl}"
MODELS="${MODELS:-gemma3-27b-base}"
TEMPERATURES="${TEMPERATURES:-1.0,0.7,1.5}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-offload/gemma3-27b}"
RESULTS_DIR="${RESULTS_DIR:-results/gemma3_27b_unquantized}"
LOGS_DIR="${LOGS_DIR:-logs/gemma3_27b_unquantized}"
RETRIES="${RETRIES:-1}"
LIMIT="${LIMIT:-}"

log() {
  printf '[goCarlos] %s\n' "$*"
}

die() {
  printf '[goCarlos] ERROR: %s\n' "$*" >&2
  exit 1
}

find_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN=$PYTHON_BIN was not found"
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - "$MIN_PYTHON" <<'PY'
import sys
need = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:2] >= need else 1)
PY
      then
        printf '%s\n' "$candidate"
        return
      fi
    fi
  done

  die "Python $MIN_PYTHON+ is required. Install python3.11+ and rerun ./goCarlos."
}

detect_cpu_memory_gib() {
  if command -v awk >/dev/null 2>&1 && [ -r /proc/meminfo ]; then
    awk '/MemTotal/ { gib=int(($2/1024/1024)-8); if (gib < 16) gib=16; print gib "GiB"; exit }' /proc/meminfo
  else
    printf '160GiB\n'
  fi
}

detect_gpu_max_memory() {
  if [ -n "${MAX_MEMORY:-}" ]; then
    printf '%s\n' "$MAX_MEMORY"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; using conservative two-GPU memory default"
    printf '0=44GiB,1=44GiB,cpu=%s\n' "$(detect_cpu_memory_gib)"
    return
  fi

  local rows entries index total reserve usable cpu_memory
  rows="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -z "$rows" ]; then
    log "Could not read GPU memory; using conservative two-GPU memory default"
    printf '0=44GiB,1=44GiB,cpu=%s\n' "$(detect_cpu_memory_gib)"
    return
  fi

  entries=""
  while IFS=, read -r index total; do
    index="${index//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    [ -n "$index" ] || continue
    [ -n "$total" ] || continue
    reserve=4096
    if [ "$total" -lt 24000 ]; then
      reserve=2048
    fi
    usable=$(( (total - reserve) / 1024 ))
    if [ "$usable" -lt 1 ]; then
      usable=1
    fi
    if [ -n "$entries" ]; then
      entries="$entries,"
    fi
    entries="${entries}${index}=${usable}GiB"
  done <<< "$rows"

  [ -n "$entries" ] || entries="0=44GiB,1=44GiB"
  cpu_memory="$(detect_cpu_memory_gib)"
  printf '%s,cpu=%s\n' "$entries" "$cpu_memory"
}

python_has_cuda_torch() {
  python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

install_torch() {
  if python_has_cuda_torch; then
    log "Found existing CUDA-capable PyTorch"
    return
  fi

  log "Installing PyTorch"
  if python -m pip install --upgrade torch; then
    if python_has_cuda_torch; then
      return
    fi
  fi

  for index_url in \
    https://download.pytorch.org/whl/cu124 \
    https://download.pytorch.org/whl/cu121 \
    https://download.pytorch.org/whl/cu118
  do
    log "Trying PyTorch wheels from $index_url"
    if python -m pip install --upgrade torch --index-url "$index_url" && python_has_cuda_torch; then
      return
    fi
  done

  die "Could not install a CUDA-capable PyTorch. Check NVIDIA drivers/CUDA visibility, then rerun ./goCarlos."
}

install_dependencies() {
  log "Installing Python dependencies"
  python -m pip install --upgrade pip setuptools wheel
  install_torch
  python -m pip install --upgrade "transformers>=4.51.0" accelerate sentencepiece protobuf safetensors huggingface_hub
}

PYTHON_BIN="$(find_python)"
MAX_MEMORY_RESOLVED="$(detect_gpu_max_memory)"

mkdir -p "$LOGS_DIR" "$OFFLOAD_FOLDER"

if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtual environment at $VENV_DIR with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || die "Could not create venv. On Ubuntu/Debian, install python3-venv."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ "$INSTALL_DEPS" = "1" ]; then
  install_dependencies
fi

if [ -z "${HF_TOKEN:-}" ]; then
  if ! python - <<'PY'
from pathlib import Path
token = Path.home() / ".cache" / "huggingface" / "token"
raise SystemExit(0 if token.exists() and token.read_text(encoding="utf-8").strip() else 1)
PY
  then
    log "HF_TOKEN is not set and no cached Hugging Face token was found."
    log "If Gemma access is gated on this machine, run: source $VENV_DIR/bin/activate && huggingface-cli login"
  fi
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
  --max-memory "$MAX_MEMORY_RESOLVED"
  --offload-folder "$OFFLOAD_FOLDER"
  --retries "$RETRIES"
  --trust-remote-code
)

if [ -n "$LIMIT" ]; then
  COMMAND+=(--limit "$LIMIT")
fi

echo "Running unquantized Gemma 3 27B WCS audit"
echo "Models: $MODELS"
echo "Dtype: $DTYPE; device_map: $DEVICE_MAP; max_memory: $MAX_MEMORY_RESOLVED"
echo "Results: $RESULTS_DIR"
echo "Logs: $LOGS_DIR"
echo

"${COMMAND[@]}" 2>&1 | tee "$LOGS_DIR/run.log"
