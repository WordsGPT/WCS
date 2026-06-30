#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PD_OUTPUT="${PD_OUTPUT:-data/processed/spanish_pd_books}"
SAMPLES="${SAMPLES:-data/processed/samples.spanish_pd_books.100x10.jsonl}"
COHERENCE_LOG="${COHERENCE_LOG:-data/processed/spanish_pd_books.coherence.jsonl}"
FREQUENCY="${FREQUENCY:-data/raw/spanish_frequency.tsv}"
FREQUENCY_SHARDS="${FREQUENCY_SHARDS:-0}"
CONTEXT_SHARDS="${CONTEXT_SHARDS:-4}"
RANK_MIN="${RANK_MIN:-10000}"
RANK_MAX="${RANK_MAX:-40000}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
CONTEXTS_PER_WORD="${CONTEXTS_PER_WORD:-10}"
CANDIDATE_CONTEXTS_PER_WORD="${CANDIDATE_CONTEXTS_PER_WORD:-40}"
COHERENCE_WORKERS="${COHERENCE_WORKERS:-4}"
COHERENCE_MODEL="${COHERENCE_MODEL:-gemini-2.5-flash-lite}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-256}"
SEED="${SEED:-13}"
PREPARE="${PREPARE:-1}"
LOCAL_ONLY="${LOCAL_ONLY:-0}"

if [[ "$PREPARE" == "1" ]]; then
  prepare_command=(
    "$PYTHON_BIN" scripts/prepare_spanish_pd_books.py
    --frequency-shards "$FREQUENCY_SHARDS"
    --context-shards "$CONTEXT_SHARDS"
    --output-dir "$PD_OUTPUT"
  )
  if [[ "$LOCAL_ONLY" == "1" ]]; then
    prepare_command+=(--local-only)
  fi
  "${prepare_command[@]}"
fi

build_command=(
  "$PYTHON_BIN" -u scripts/build_samples.py
  --frequency "$FREQUENCY"
  --corpus "$PD_OUTPUT/contexts"
  --output "$SAMPLES"
  --rank-min "$RANK_MIN"
  --rank-max "$RANK_MAX"
  --sample-size "$SAMPLE_SIZE"
  --contexts-per-word "$CONTEXTS_PER_WORD"
  --candidate-contexts-per-word "$CANDIDATE_CONTEXTS_PER_WORD"
  --coherence-workers "$COHERENCE_WORKERS"
  --coherence-model "$COHERENCE_MODEL"
  --coherence-log "$COHERENCE_LOG"
  --context-tokens "$CONTEXT_TOKENS"
  --seed "$SEED"
  --min-word-length 4
  --language Spanish
  --validate-target-words
  --exclude-capitalized-matches
  --progress-interval 1
  --require-full-sample
)
if [[ -s "$SAMPLES" ]]; then
  build_command+=(--resume)
fi
"${build_command[@]}"
