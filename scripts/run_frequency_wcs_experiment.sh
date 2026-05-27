#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
FREQUENCY="${FREQUENCY:-data/raw/norvig_count_1w.txt}"
CORPUS="${CORPUS:-}"
SAMPLES="${SAMPLES:-data/processed/samples.frequency_1k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/frequency_wcs}"
LOG_DIR="${LOG_DIR:-logs/frequency_wcs}"
STRATA="${STRATA:-1:1000:200,1001:10000:400,10001:100000:400}"
MODELS="${MODELS:-gemma2-9b-base,gemma3-12b-it,llama31-8b-instruct,mistral7b-v03-base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CONTEXTS_PER_WORD="${CONTEXTS_PER_WORD:-10}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-256}"
SEED="${SEED:-20260527}"
MIN_WORD_LENGTH="${MIN_WORD_LENGTH:-3}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
TOKEN_PLOT_WORD_LIMIT="${TOKEN_PLOT_WORD_LIMIT:-100}"
SKIP_COHERENCE_CHECK="${SKIP_COHERENCE_CHECK:-1}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$FORCE_REBUILD" == "1" || ! -s "$SAMPLES" ]]; then
  if [[ -z "$CORPUS" ]]; then
    echo "Set CORPUS to a local text corpus path before building samples." >&2
    echo "Example: CORPUS=/data/pg19/test $0" >&2
    echo "If $SAMPLES already exists, CORPUS is not needed." >&2
    exit 2
  fi
  build_command=(
    "$PYTHON_BIN" -u scripts/build_stratified_frequency_samples.py
    --frequency "$FREQUENCY"
    --corpus "$CORPUS"
    --output "$SAMPLES"
    --strata "$STRATA"
    --contexts-per-word "$CONTEXTS_PER_WORD"
    --context-tokens "$CONTEXT_TOKENS"
    --seed "$SEED"
    --min-word-length "$MIN_WORD_LENGTH"
    --exclude-capitalized-matches
    --resume
  )
  if [[ "$SKIP_COHERENCE_CHECK" == "1" ]]; then
    build_command+=(--skip-coherence-check)
  fi
  "${build_command[@]}"
else
  echo "[samples] using existing $SAMPLES"
fi

"$PYTHON_BIN" -u scripts/run_model_suite.py \
  --samples "$SAMPLES" \
  --results-dir "$OUTPUT_DIR" \
  --logs-dir "$LOG_DIR/model_suite" \
  --models "$MODELS" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

audit_paths=("$OUTPUT_DIR"/audit.*.jsonl)
if [[ ! -e "${audit_paths[0]}" ]]; then
  echo "No audit JSONL files found in $OUTPUT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_wcs.py \
  --audits "${audit_paths[@]}" \
  --summary "$OUTPUT_DIR/wcs_summary.csv"

"$PYTHON_BIN" scripts/summarize_wcs_words.py \
  --audits "${audit_paths[@]}" \
  --summary "$OUTPUT_DIR/wcs_word_summary.csv"

"$PYTHON_BIN" scripts/summarize_frequency_wcs.py \
  --samples "$SAMPLES" \
  --per-word-output "$OUTPUT_DIR/per_word_wcs.csv" \
  --correlation-output "$OUTPUT_DIR/frequency_correlations.csv" \
  --plot-dir "$OUTPUT_DIR/plots" \
  --token-plot-word-limit "$TOKEN_PLOT_WORD_LIMIT" \
  "${audit_paths[@]}"

echo "[done] frequency WCS outputs are in $OUTPUT_DIR"
