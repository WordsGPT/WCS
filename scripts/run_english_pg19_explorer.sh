#!/usr/bin/env bash
set -euo pipefail

# One-command, resumable English PG-19 audit and explorer build.
# Usage:
#   ./scripts/run_english_pg19_explorer.sh start
#   ./scripts/run_english_pg19_explorer.sh status
#   ./scripts/run_english_pg19_explorer.sh logs
#   ./scripts/run_english_pg19_explorer.sh run    # foreground

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
[ "$SCRIPT_DIR" != "${BASH_SOURCE[0]}" ] || SCRIPT_DIR="."
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

ACTION="${1:-start}"
VENV_DIR="${VENV_DIR:-.venv-english-pg19}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
MIN_PYTHON="${MIN_PYTHON:-3.11}"
SAMPLES="${SAMPLES:-data/processed/samples.jsonl}"
MODELS="${MODELS:-english-pg19}"
TOP_K="${TOP_K:-5}"
RANK_NEIGHBORS="${RANK_NEIGHBORS:-5}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESULTS_DIR="${RESULTS_DIR:-results/english_pg19_predictions}"
LOGS_DIR="${LOGS_DIR:-logs/english_pg19_predictions}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-offload/english_pg19_predictions}"
EXPLORER_DATA="${EXPLORER_DATA:-explorer_data.english.json}"
RETRIES="${RETRIES:-1}"
LIMIT="${LIMIT:-}"
PID_FILE="$LOGS_DIR/runner.pid"
MASTER_LOG="$LOGS_DIR/runner.log"

if [ -n "$LIMIT" ]; then
  RESULTS_DIR="${RESULTS_DIR}.smoke-${LIMIT}"
  LOGS_DIR="${LOGS_DIR}.smoke-${LIMIT}"
  OFFLOAD_FOLDER="${OFFLOAD_FOLDER}.smoke-${LIMIT}"
  EXPLORER_DATA="${EXPLORER_DATA%.json}.smoke-${LIMIT}.json"
  PID_FILE="$LOGS_DIR/runner.pid"
  MASTER_LOG="$LOGS_DIR/runner.log"
fi

log() {
  printf '[english-pg19] %s\n' "$*"
}

die() {
  printf '[english-pg19] ERROR: %s\n' "$*" >&2
  exit 1
}

find_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN=$PYTHON_BIN was not found"
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" - "$MIN_PYTHON" <<'PY'
import sys
required = tuple(int(part) for part in sys.argv[1].split("."))
raise SystemExit(0 if sys.version_info[:2] >= required else 1)
PY
    then
      printf '%s\n' "$candidate"
      return
    fi
  done
  die "Python $MIN_PYTHON+ is required (including the venv module)."
}

cuda_torch_is_usable() {
  python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
ok = version >= (2, 4) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
raise SystemExit(0 if ok else 1)
PY
}

reported_cuda_version() {
  nvidia-smi 2>/dev/null |
    sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' |
    head -n 1
}

version_at_least() {
  "$BASE_PYTHON" - "$1" "$2" <<'PY'
import sys
actual = tuple(int(part) for part in sys.argv[1].split("."))
required = tuple(int(part) for part in sys.argv[2].split("."))
raise SystemExit(0 if actual >= required else 1)
PY
}

install_torch() {
  if cuda_torch_is_usable; then
    log "Keeping the existing CUDA/bfloat16-capable PyTorch installation"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi is unavailable; install/enable the NVIDIA driver first"

  local cuda_version
  cuda_version="$(reported_cuda_version)"
  [ -n "$cuda_version" ] || die "Could not determine the driver-supported CUDA version"
  log "Installing PyTorch for driver-reported CUDA $cuda_version"

  if version_at_least "$cuda_version" "13.0"; then
    python -m pip install --upgrade "torch==2.12.1"
  elif version_at_least "$cuda_version" "12.6"; then
    python -m pip install --upgrade "torch==2.12.1" \
      --index-url https://download.pytorch.org/whl/cu126
  elif version_at_least "$cuda_version" "11.8"; then
    log "Using the CUDA 11.8 compatibility build of PyTorch 2.7.1"
    python -m pip install --upgrade "torch==2.7.1" \
      --index-url https://download.pytorch.org/whl/cu118
  else
    die "CUDA $cuda_version is too old; update the NVIDIA driver to CUDA 11.8+"
  fi
  cuda_torch_is_usable ||
    die "Installed PyTorch cannot use CUDA+bfloat16; inspect nvidia-smi and the driver"
}

hf_auth_is_ready() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  command -v hf >/dev/null 2>&1 || return 1
  hf auth whoami >/dev/null 2>&1
}

ensure_hf_auth() {
  if hf_auth_is_ready; then
    return
  fi
  if [ -t 0 ]; then
    command -v hf >/dev/null 2>&1 ||
      die "The Hugging Face CLI is unavailable; rerun with INSTALL_DEPS=1"
    log "Hugging Face login is required for gated Llama and Gemma checkpoints"
    hf auth login
    hf_auth_is_ready || die "Hugging Face authentication failed"
    return
  fi
  die "No Hugging Face login found. Run interactively once or provide HF_TOKEN."
}

detect_cpu_memory_gib() {
  awk '/MemTotal/ { value=int(($2/1024/1024)-8); if (value < 16) value=16; print value "GiB"; exit }' \
    /proc/meminfo
}

detect_max_memory() {
  if [ -n "${MAX_MEMORY:-}" ]; then
    printf '%s\n' "$MAX_MEMORY"
    return
  fi
  local rows entries index total reserve usable
  rows="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null)"
  [ -n "$rows" ] || die "Could not query GPU memory with nvidia-smi"
  entries=""
  while IFS=, read -r index total; do
    index="${index//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    reserve=4096
    [ "$total" -ge 24000 ] || reserve=2048
    usable=$(( (total - reserve) / 1024 ))
    [ "$usable" -ge 1 ] || usable=1
    [ -z "$entries" ] || entries="${entries},"
    entries="${entries}${index}=${usable}GiB"
  done <<< "$rows"
  printf '%s,cpu=%s\n' "$entries" "$(detect_cpu_memory_gib)"
}

setup_environment() {
  BASE_PYTHON="$(find_python)"
  export BASE_PYTHON
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating $VENV_DIR"
    "$BASE_PYTHON" -m venv "$VENV_DIR" ||
      die "Could not create the venv (install python3-venv on Debian/Ubuntu)"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  if [ "$INSTALL_DEPS" = "1" ]; then
    python -m pip install --upgrade pip wheel
    install_torch
    python -m pip install --upgrade -r requirements.english-pg19-gpu.txt
  fi
  cuda_torch_is_usable ||
    die "The venv does not contain a CUDA/bfloat16-capable PyTorch 2.4+"
  ensure_hf_auth
}

run_preflight() {
  local command=(python scripts/preflight_english_pg19.py --samples "$SAMPLES" --models "$MODELS")
  [ -z "$LIMIT" ] || command+=(--limit "$LIMIT")
  "${command[@]}"
}

run_worker() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$OFFLOAD_FOLDER"
  trap 'rm -f "$PID_FILE"' EXIT
  local max_memory
  max_memory="$(detect_max_memory)"
  log "Starting resumable audit: models=$MODELS, max_memory=$max_memory"
  local command=(
    python -u scripts/run_model_suite.py
    --samples "$SAMPLES"
    --models "$MODELS"
    --temperature 1.0
    --top-k "$TOP_K"
    --rank-neighbors "$RANK_NEIGHBORS"
    --results-dir "$RESULTS_DIR"
    --logs-dir "$LOGS_DIR/models"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --device-map "$DEVICE_MAP"
    --max-memory "$max_memory"
    --offload-folder "$OFFLOAD_FOLDER"
    --retries "$RETRIES"
  )
  [ -z "$LIMIT" ] || command+=(--limit "$LIMIT")
  "${command[@]}"

  python scripts/build_explorer_data.py \
    --samples "$SAMPLES" \
    --audits "$RESULTS_DIR/audit.*.jsonl" \
    --output "$EXPLORER_DATA" \
    --dataset "English PG-19" \
    --language English \
    --temperature 1.0
  log "Audit and explorer build completed: $EXPLORER_DATA"
}

is_running() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

show_status() {
  if is_running; then
    log "Running as PID $(cat "$PID_FILE")"
  else
    log "Not running"
  fi
  if [ -f "$RESULTS_DIR/manifest.json" ]; then
    python - "$RESULTS_DIR/manifest.json" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
done = [row["slug"] for row in rows if row.get("completed")]
print(f"[english-pg19] Completed models: {len(done)}/{len(rows)}")
if done:
    print("[english-pg19] " + ", ".join(done))
PY
  fi
  [ ! -f "$MASTER_LOG" ] || tail -n 20 "$MASTER_LOG"
}

case "$ACTION" in
  setup)
    setup_environment
    run_preflight
    log "Setup and preflight completed"
    ;;
  run)
    setup_environment
    run_preflight
    run_worker
    ;;
  worker)
    run_worker
    ;;
  start)
    mkdir -p "$LOGS_DIR"
    if is_running; then
      die "A run is already active as PID $(cat "$PID_FILE")"
    fi
    setup_environment
    run_preflight
    log "Launching the audit in the background"
    nohup bash "$0" worker >>"$MASTER_LOG" 2>&1 &
    worker_pid=$!
    printf '%s\n' "$worker_pid" >"$PID_FILE"
    log "Started PID $worker_pid"
    log "Monitor with: bash $0 status"
    log "Follow logs with: bash $0 logs"
    ;;
  status)
    show_status
    ;;
  logs)
    [ -f "$MASTER_LOG" ] || die "No master log exists at $MASTER_LOG"
    tail -f "$MASTER_LOG"
    ;;
  *)
    die "Unknown action '$ACTION'. Use setup, start, run, status, logs, or worker."
    ;;
esac
