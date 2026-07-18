#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-fineweb_200x50}"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON="$VENV/bin/python"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/$RUN_ID/wcs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/$RUN_ID/wcs_temperature_extension}"
NEW_TEMPERATURES="${NEW_TEMPERATURES:-0.6,0.7,1.5}"
ALL_TEMPERATURES="${ALL_TEMPERATURES:-0.6,0.7,1.0,1.5}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-10000}"

MODELS="${MODELS:-llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,qwen35-9b-base,qwen35-9b-instruct,qwen25-14b-base,qwen25-14b-instruct,gemma3-12b-base,gemma3-12b-it,gemma4-e4b-base,gemma4-e4b-it,deepseek-qwen14b-distill}"

if [[ -n "${SAMPLES:-}" ]]; then
  SAMPLES_PATH="$SAMPLES"
elif [[ -f "$ROOT/samples.${RUN_ID}.jsonl" ]]; then
  SAMPLES_PATH="$ROOT/samples.${RUN_ID}.jsonl"
else
  SAMPLES_PATH="$ROOT/data/processed/samples.${RUN_ID}.jsonl"
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
exec > >(tee -a "$LOG_DIR/launcher.log") 2>&1
trap 'code=$?; echo "[error] line ${LINENO}; exit ${code}; see $LOG_DIR/launcher.log"' ERR

echo "[start] $(date --iso-8601=seconds) FineWeb temperature extension"
echo "[plan] new temperatures=$NEW_TEMPERATURES; reused baseline=T=1.0"
echo "[plan] models=$MODELS"
echo "[output] results=$RESULTS_DIR logs=$LOG_DIR"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[note] no Hugging Face token is exported; continuing with the saved login and local model cache"
fi
if [[ ! -f "$SAMPLES_PATH" ]]; then
  echo "[error] FineWeb sample file not found at $SAMPLES_PATH"
  exit 2
fi

SAMPLE_COUNT="$(wc -l < "$SAMPLES_PATH")"
if [[ "$SAMPLE_COUNT" -ne "$EXPECTED_SAMPLES" ]]; then
  echo "[error] expected $EXPECTED_SAMPLES samples; found $SAMPLE_COUNT in $SAMPLES_PATH"
  exit 2
fi

echo "[setup] creating/updating the isolated environment and running tests"
PYTHON_BIN="$PYTHON_BIN" VENV="$VENV" LOG_DIR="$LOG_DIR/setup" \
  "$ROOT/scripts/setup_overnight.sh"

echo "[check] validating the existing T=1 audits; no inference is allowed here"
"$PYTHON" -u scripts/run_model_suite.py \
  --samples "$SAMPLES_PATH" \
  --models "$MODELS" \
  --temperatures 1.0 \
  --index-only \
  --results-dir "$RESULTS_DIR" \
  --logs-dir "$LOG_DIR" \
  --summary "$LOG_DIR/wcs_summary.t1_check.csv" \
  --word-summary "$LOG_DIR/wcs_word_summary.t1_check.csv"

echo "[audit] running only the added temperatures in one model forward pass"
"$PYTHON" -u scripts/run_model_suite.py \
  --samples "$SAMPLES_PATH" \
  --models "$MODELS" \
  --temperatures "$NEW_TEMPERATURES" \
  --results-dir "$RESULTS_DIR" \
  --logs-dir "$LOG_DIR" \
  --summary "$RESULTS_DIR/wcs_summary.temperature_extension.csv" \
  --word-summary "$RESULTS_DIR/wcs_word_summary.temperature_extension.csv" \
  --top-k 64 \
  --dtype bfloat16 \
  --trust-remote-code

echo "[index] validating all four temperatures and rebuilding combined summaries"
"$PYTHON" -u scripts/run_model_suite.py \
  --samples "$SAMPLES_PATH" \
  --models "$MODELS" \
  --temperatures "$ALL_TEMPERATURES" \
  --index-only \
  --results-dir "$RESULTS_DIR" \
  --logs-dir "$LOG_DIR" \
  --summary "$RESULTS_DIR/wcs_summary.csv" \
  --word-summary "$RESULTS_DIR/wcs_word_summary.csv"

AUDITS=()
IFS=',' read -r -a TEMPERATURE_LIST <<< "$NEW_TEMPERATURES"
for temperature in "${TEMPERATURE_LIST[@]}"; do
  clean_temperature="${temperature//[[:space:]]/}"
  normalized_temperature="$(awk -v value="$clean_temperature" 'BEGIN { printf "%g", value }')"
  temperature_slug="t${normalized_temperature//./p}"
  while IFS= read -r -d '' audit; do
    AUDITS+=("$audit")
  done < <(find "$RESULTS_DIR/$temperature_slug" -maxdepth 1 -type f \
    -name 'audit.*.jsonl' -print0 | sort -z)
done

echo "[recipes] computing sequential top-k then top-p filters for added temperatures"
"$PYTHON" -u scripts/summarize_combined_filters.py \
  --audits "${AUDITS[@]}" \
  --recipe gemma_topk64_topp095:64:0.95 \
  --recipe qwen_topk20_topp095:20:0.95 \
  --recipe qwen_topk20_topp080:20:0.80 \
  --summary "$RESULTS_DIR/wcs_summary.combined_recipes.csv" \
  --word-summary "$RESULTS_DIR/wcs_word_summary.combined_recipes.csv"

echo "[done] $(date --iso-8601=seconds) results=$RESULTS_DIR"
echo "[resume] rerun this same command after any interruption"
