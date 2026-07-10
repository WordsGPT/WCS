#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
RUN_ID="${RUN_ID:-fineweb_200x50}"
SAMPLES="${SAMPLES:-$ROOT/data/processed/samples.${RUN_ID}.jsonl}"
CANDIDATES="${CANDIDATES:-$ROOT/data/processed/candidates.${RUN_ID}.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/$RUN_ID}"
MODELS="${MODELS:-llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,gemma3-12b-base,gemma3-12b-it}"
N_WORDS="${N_WORDS:-200}"
CONTEXTS_PER_WORD="${CONTEXTS_PER_WORD:-50}"
GEN_CONTEXTS="${GEN_CONTEXTS:-50}"
GEN_WORDS="${GEN_WORDS:-200}"
COHERENCE_WORKERS="${COHERENCE_WORKERS:-12}"
TEMPERATURES="${TEMPERATURES:-1.0}"
SEED="${SEED:-13}"
MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-140}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
exec > >(tee -a "$LOG_DIR/overnight.log") 2>&1
trap 'code=$?; echo "[error] line ${LINENO}; exit ${code}; see $LOG_DIR/overnight.log"' ERR

if [[ ! -x "$PYTHON" ]]; then
  echo "[error] .venv is missing. Run scripts/setup_overnight.sh first."
  exit 2
fi

echo "[start] $(date --iso-8601=seconds) run=$RUN_ID models=$MODELS"
echo "[plan] WCS: ${N_WORDS} words x ${CONTEXTS_PER_WORD} contexts; generation: ${GEN_CONTEXTS} contexts x ${GEN_WORDS} words"

"$PYTHON" -u scripts/preflight_overnight.py \
  --models "$MODELS" \
  --min-free-disk-gib "$MIN_FREE_DISK_GIB"

# Load every architecture and run one real CUDA forward pass now, so gating,
# OOM, and Transformers compatibility failures appear before data preparation.
"$PYTHON" -u scripts/run_model_suite.py \
  --samples data/processed/samples.jsonl \
  --models "$MODELS" \
  --limit 1 \
  --results-dir "$RESULTS_DIR/preflight_models" \
  --logs-dir "$LOG_DIR/preflight_models" \
  --dtype bfloat16 \
  --trust-remote-code \
  --retries 0

"$PYTHON" -u scripts/prepare_fineweb_samples.py \
  --frequency data/raw/norvig_count_1w.txt \
  --output "$SAMPLES" \
  --candidates "$CANDIDATES" \
  --coherence-log "$LOG_DIR/coherence.jsonl" \
  --sample-size "$N_WORDS" \
  --contexts-per-word "$CONTEXTS_PER_WORD" \
  --candidate-contexts-per-word "$((CONTEXTS_PER_WORD + 30))" \
  --coherence-workers "$COHERENCE_WORKERS" \
  --seed "$SEED" \
  --resume

"$PYTHON" -u scripts/run_model_suite.py \
  --samples "$SAMPLES" \
  --models "$MODELS" \
  --temperatures "$TEMPERATURES" \
  --results-dir "$RESULTS_DIR/wcs" \
  --logs-dir "$LOG_DIR/wcs" \
  --dtype bfloat16 \
  --trust-remote-code

"$PYTHON" -u scripts/run_open_generation.py \
  --samples "$SAMPLES" \
  --models "$MODELS" \
  --contexts "$GEN_CONTEXTS" \
  --words "$GEN_WORDS" \
  --results-dir "$RESULTS_DIR/generation" \
  --dtype bfloat16 \
  --trust-remote-code \
  --resume

echo "[done] $(date --iso-8601=seconds) results=$RESULTS_DIR"
