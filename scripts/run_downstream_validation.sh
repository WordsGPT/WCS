#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
RUN_ID="${RUN_ID:-fineweb_200x50}"
if [[ -f "$ROOT/samples.${RUN_ID}.jsonl" ]]; then
  DEFAULT_SAMPLES="$ROOT/samples.${RUN_ID}.jsonl"
else
  DEFAULT_SAMPLES="$ROOT/data/processed/samples.${RUN_ID}.jsonl"
fi
SAMPLES="${SAMPLES:-$DEFAULT_SAMPLES}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/$RUN_ID/downstream_validation}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/$RUN_ID/downstream_validation}"

# Full 13-model downstream validation set
MODELS="${MODELS:-llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,qwen35-9b-base,qwen35-9b-instruct,qwen25-14b-base,qwen25-14b-instruct,gemma3-12b-base,gemma3-12b-it,gemma4-e4b-base,gemma4-e4b-it,deepseek-qwen14b-distill}"

CONTEXTS="${CONTEXTS:-50}"
WORDS="${WORDS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURES="${TEMPERATURES:-0.7,1.0}"
TOP_K_VALUES="${TOP_K_VALUES:-10,20,50,80}"
TOP_P_VALUES="${TOP_P_VALUES:-0.80,0.90,0.95,0.99}"
MIN_P_VALUES="${MIN_P_VALUES:-0.01,0.05,0.10}"
SEED="${SEED:-13}"

# Optional mode flags: set WCS_ONLY=1 or GENERATION_ONLY=1
WCS_ONLY="${WCS_ONLY:-0}"
GENERATION_ONLY="${GENERATION_ONLY:-0}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
trap 'code=$?; echo "[error] line ${LINENO}; exit ${code}; see $LOG_DIR/run.log"' ERR

if [[ ! -x "$PYTHON" ]]; then
  echo "[error] Python environment not found at $PYTHON. Run scripts/setup_overnight.sh once."
  exit 2
fi
if [[ ! -f "$SAMPLES" ]]; then
  echo "[error] FineWeb samples not found at $SAMPLES"
  exit 2
fi

echo "[start] $(date --iso-8601=seconds) downstream validation"
echo "[plan] models=$MODELS contexts=$CONTEXTS words=$WORDS temperatures=$TEMPERATURES"
echo "[plan] top_k=$TOP_K_VALUES top_p=$TOP_P_VALUES min_p=$MIN_P_VALUES plus untruncated"
echo "[plan] wcs_only=$WCS_ONLY generation_only=$GENERATION_ONLY"

MODE_FLAGS=()
if [[ "$WCS_ONLY" == "1" ]]; then
  MODE_FLAGS+=(--wcs-only)
fi
if [[ "$GENERATION_ONLY" == "1" ]]; then
  MODE_FLAGS+=(--generation-only)
fi

"$PYTHON" -u scripts/run_downstream_validation.py \
  --samples "$SAMPLES" \
  --results-dir "$RESULTS_DIR" \
  --models "$MODELS" \
  --contexts "$CONTEXTS" \
  --words "$WORDS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperatures "$TEMPERATURES" \
  --top-k-values "$TOP_K_VALUES" \
  --top-p-values "$TOP_P_VALUES" \
  --min-p-values "$MIN_P_VALUES" \
  --dtype bfloat16 \
  --trust-remote-code \
  --resume \
  "${MODE_FLAGS[@]+${MODE_FLAGS[@]}}"

echo "[done] $(date --iso-8601=seconds) results=$RESULTS_DIR"
