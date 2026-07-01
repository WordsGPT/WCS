#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-data/processed/samples.spanish_pd_books.100x10.jsonl}"
INDEX="${INDEX:-data/processed/samples.spanish_pd_books.100x10.index.json}"
REPAIRED_SAMPLES="${REPAIRED_SAMPLES:-data/processed/samples.spanish_pd_books.100x10.repaired.jsonl}"
REAUDIT_LOG="${REAUDIT_LOG:-data/processed/spanish_pd_books.reaudit.jsonl}"
COHERENCE_MODEL="${COHERENCE_MODEL:-gemini-2.5-flash}"
COHERENCE_WORKERS="${COHERENCE_WORKERS:-1}"
GEMINI_MAX_ATTEMPTS="${GEMINI_MAX_ATTEMPTS:-8}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
RESULTS_DIR="${RESULTS_DIR:-results/spanish_pd_books_repaired}"
LOGS_DIR="${LOGS_DIR:-logs/spanish_pd_books_repaired}"

"$PYTHON_BIN" -u scripts/repair_spanish_pd_contexts.py \
  --input "$INPUT" \
  --index "$INDEX" \
  --output "$REPAIRED_SAMPLES" \
  --audit-log "$REAUDIT_LOG" \
  --model "$COHERENCE_MODEL" \
  --workers "$COHERENCE_WORKERS" \
  --max-attempts "$GEMINI_MAX_ATTEMPTS" \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE"

suite_command=(
  "$PYTHON_BIN" -u scripts/run_model_suite.py
  --samples "$REPAIRED_SAMPLES"
  --results-dir "$RESULTS_DIR"
  --logs-dir "$LOGS_DIR"
)
if [[ -n "${MODELS:-}" ]]; then
  suite_command+=(--models "$MODELS")
fi
if [[ -n "${TEMPERATURES:-}" ]]; then
  suite_command+=(--temperatures "$TEMPERATURES")
fi
"${suite_command[@]}"
