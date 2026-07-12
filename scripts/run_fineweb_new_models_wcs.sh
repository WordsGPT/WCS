#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-$VENV/bin/python}"
RUN_ID="${RUN_ID:-fineweb_200x50}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/$RUN_ID/wcs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/$RUN_ID/wcs_new_models}"
TEMPERATURES="${TEMPERATURES:-1.0}"

NEW_MODELS_DEFAULT="qwen35-9b-base,qwen35-9b-instruct,qwen25-14b-base,qwen25-14b-instruct,gemma4-e4b-base,gemma4-e4b-it,deepseek-qwen14b-distill"
ALL_MODELS="llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,gemma3-12b-base,gemma3-12b-it,$NEW_MODELS_DEFAULT"
MODELS="${MODELS:-$NEW_MODELS_DEFAULT}"

if [[ -n "${SAMPLES:-}" ]]; then
  SAMPLES_PATH="$SAMPLES"
elif [[ -f "$ROOT/samples.fineweb_200x50.jsonl" ]]; then
  SAMPLES_PATH="$ROOT/samples.fineweb_200x50.jsonl"
else
  SAMPLES_PATH="$ROOT/data/processed/samples.fineweb_200x50.jsonl"
fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1
trap 'code=$?; echo "[error] line ${LINENO}; exit ${code}; see $LOG_DIR/launcher.log"' ERR

if [[ ! -x "$PYTHON" ]]; then
  echo "[error] Python environment not found at $PYTHON"
  exit 2
fi
if [[ ! -f "$SAMPLES_PATH" ]]; then
  echo "[error] FineWeb sample file not found at $SAMPLES_PATH"
  exit 2
fi

SAMPLE_COUNT="$(wc -l < "$SAMPLES_PATH")"
if [[ "$SAMPLE_COUNT" -ne 10000 ]]; then
  echo "[error] expected 10,000 FineWeb samples; found $SAMPLE_COUNT in $SAMPLES_PATH"
  exit 2
fi

echo "[start] $(date --iso-8601=seconds)"
echo "[plan] samples=$SAMPLES_PATH models=$MODELS temperatures=$TEMPERATURES"
echo "[output] results=$RESULTS_DIR logs=$LOG_DIR"

"$PYTHON" -u scripts/run_model_suite.py \
  --samples "$SAMPLES_PATH" \
  --models "$MODELS" \
  --temperatures "$TEMPERATURES" \
  --results-dir "$RESULTS_DIR" \
  --logs-dir "$LOG_DIR" \
  --summary "$RESULTS_DIR/wcs_summary.new_models.csv" \
  --word-summary "$RESULTS_DIR/wcs_word_summary.new_models.csv" \
  --manifest "$RESULTS_DIR/manifest.new_models.json" \
  --dtype bfloat16 \
  --trust-remote-code

# Re-run the suite index over all 13 reviewer-site checkpoints. Completed
# audits are skipped, and the combined summaries are rebuilt from every raw
# audit rather than by concatenating per-server CSV files.
if [[ "$MODELS" == "$NEW_MODELS_DEFAULT" && "$TEMPERATURES" == "1.0" ]]; then
  "$PYTHON" -u scripts/run_model_suite.py \
    --samples "$SAMPLES_PATH" \
    --models "$ALL_MODELS" \
    --temperatures "$TEMPERATURES" \
    --results-dir "$RESULTS_DIR" \
    --logs-dir "$LOG_DIR" \
    --summary "$RESULTS_DIR/wcs_summary.csv" \
    --word-summary "$RESULTS_DIR/wcs_word_summary.csv" \
    --manifest "$RESULTS_DIR/manifest.json" \
    --dtype bfloat16 \
    --trust-remote-code \
    --retries 0
else
  echo "[note] custom MODELS/TEMPERATURES used; combined 13-model summary was not rebuilt"
fi

echo "[done] $(date --iso-8601=seconds) results=$RESULTS_DIR"
