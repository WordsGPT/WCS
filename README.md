# Word Coverage Score

This repository contains code for the Word Coverage Score (WCS) paper.

The current implementation covers Phase 1 and Phase 3:

- Phase 1: project scaffold and data contracts.
- Phase 3: dataset builder for middle-long-tail word sampling and PG-19-style context extraction.

## Data Flow

1. Load a ranked frequency list.
2. Randomly order words in the paper's middle-long-tail rank band.
3. Search a local corpus for each selected word.
4. Extract the preceding context window.
5. Continue through the rank band until the requested number of usable words is reached.
6. Keep the requested number of Gemini-checked contexts for each selected word.
7. Write a stable `samples.jsonl` file for later forced-path model audits.

See [docs/data_contracts.md](docs/data_contracts.md) for the file formats.

## Build Samples

Example using the bundled fixtures:

```bash
python scripts/build_samples.py \
  --frequency tests/fixtures/frequency/norvig_sample.tsv \
  --corpus tests/fixtures/pg19_sample \
  --output data/processed/samples.fixture.jsonl \
  --rank-min 3 \
  --rank-max 8 \
  --sample-size 4 \
  --contexts-per-word 10 \
  --context-tokens 16 \
  --seed 7 \
  --min-word-length 3 \
  --dictionary /usr/share/dict/linux.words \
  --exclude-capitalized-matches
```

For PG-19, point `--corpus` at a directory containing `.txt` files from the test partition.

`--sample-size` is the number of target words. `--contexts-per-word` is the
number of accepted text contexts kept for each word. The builder first filters
out words without enough raw corpus contexts, then submits several candidate
contexts for each word to Gemini in one structured request. By default it uses
`gemini-2.5-flash-lite`, sends 40 candidates per word, and runs four word-level
requests concurrently. Adjust `--candidate-contexts-per-word` and
`--coherence-workers` to match the API quota.

Use `--resume` to continue appending to an existing output file. Resume preserves complete word groups and continues sample IDs from the checkpoint.

Use `--exclude-capitalized-matches` for the paper dataset to reduce proper names and place names such as capitalized mythological, personal, or geographic references.

Use `--min-word-length 3` or higher to remove short frequency-list artifacts such as two-letter abbreviations.

Use `--dictionary /usr/share/dict/linux.words` to keep only targets found in the local English word list.

Use `--language Spanish` when building Spanish corpora so the Gemini coherence
check asks for coherent Spanish prose. The tokenizer and frequency-list parser
accept accented Unicode words.

Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment or `.env`. The
validator requires an API key unless `--skip-coherence-check` is explicitly
used. It checks both the quality of the excerpt and whether the target word is
a natural grammatical and semantic continuation.

## Spanish PD Books Experiment

Install the optional Parquet dependencies:

```bash
python -m pip install -e '.[pd-books]'
```

Use the existing Leipzig Spanish News 2023 frequency list with held-out
Spanish-PD-Books contexts, producing 100 words with 10 Gemini-approved contexts
each:

```bash
scripts/build_spanish_pd_books_samples.sh
```

The default preparation downloads four numerically sorted Spanish-PD-Books
Parquet shards for contexts. Target ranks come from
`data/raw/spanish_frequency.tsv`, generated from Leipzig
`spa_news_2023_1M`. It writes:

- `data/processed/spanish_pd_books/contexts/`
- `data/processed/spanish_pd_books/manifest.json`
- `data/processed/samples.spanish_pd_books.100x10.jsonl`

Existing local Parquet files can be used without network access:

```bash
LOCAL_ONLY=1 scripts/build_spanish_pd_books_samples.sh
```

The main settings can be overridden as environment variables, for example:

```bash
COHERENCE_WORKERS=8 \
CANDIDATE_CONTEXTS_PER_WORD=24 \
CONTEXT_SHARDS=6 \
scripts/build_spanish_pd_books_samples.sh
```

To derive frequencies from separate PD Books shards instead, set
`FREQUENCY_SHARDS` and point `FREQUENCY` at the generated file:

```bash
FREQUENCY_SHARDS=2 \
FREQUENCY=data/processed/spanish_pd_books/frequency.tsv \
scripts/build_spanish_pd_books_samples.sh
```

## Build Word Frequency Lists

Create ranked TSV frequency lists from local text, CSV, JSON, or lyric trees:

```bash
python scripts/build_word_frequency.py \
  lyrics \
  --output-dir data/processed/wordlists/lyrics \
  --combined-name all_lyrics \
  --min-word-length 3
```

For the Spanish Fortunata y Jacinta source:

```bash
python scripts/build_word_frequency.py \
  FortunayJacinta.txt \
  --group-name fortunayjacinta \
  --output-dir data/processed/wordlists \
  --min-word-length 4 \
  --strip-gutenberg
```

For an external Spanish list comparable to the English Norvig list, download
and convert the Leipzig Corpora Collection Spanish News 2023 word list:

```bash
python scripts/download_leipzig_frequency.py \
  --corpus spa_news_2023_1M \
  --output data/raw/spanish_frequency.tsv
```

## Notes

The context window is stored as whitespace-delimited text for corpus preparation. The later model-audit phase should re-tokenize the `prefix` and `word` with the evaluated model's tokenizer before running the forced-path audit.

## Verify

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Forced-Path Audit

Phase 2 code is available in [src/wcs/audit.py](src/wcs/audit.py).

Notebook usage is documented in [docs/notebook_usage.md](docs/notebook_usage.md).

CLI example:

```bash
python scripts/run_audit.py \
  --samples data/processed/samples.jsonl \
  --output results/audit.gpt2.smoke.jsonl \
  --model gpt2 \
  --device cuda \
  --dtype float16 \
  --limit 5
```

## WCS Aggregation

After one or more audit JSONL files exist:

```bash
python scripts/summarize_wcs.py \
  --audits results/audit.MODEL.jsonl \
  --summary results/wcs_summary.csv \
  --plot-dir plots
```

The summary CSV can be generated without plotting dependencies. Plot generation requires `matplotlib`.

## Nemotron Tri-Mode Smoke Test

NVIDIA's `nvidia/Nemotron-Labs-Diffusion-14B` exposes three inference modes:
autoregressive, diffusion/dLM, and linear self-speculation. Use the dedicated
runner to compare those execution modes on the same prompt:

```bash
python scripts/run_nemotron_tri_mode_test.py \
  --model nvidia/Nemotron-Labs-Diffusion-14B \
  --mode all \
  --max-new-tokens 128 \
  --block-length 32 \
  --threshold 0.9 \
  --dtype bfloat16 \
  --output results/nemotron_tri_mode.jsonl
```

This is separate from the WCS forced-path audit. WCS summarizes top-k, top-p,
and min-p token reachability from next-token logits; the Nemotron runner tests
the model's AR, diffusion, and self-speculation generation APIs directly.

## Resumable Model Suite

For long unattended GPU runs, use the resumable suite runner. It runs one
model at a time, writes each audit to a `.partial` file first, renames it only
after validation, skips already completed outputs, retries failures, and
rewrites the combined context-level and word-level summaries after every
completed model.

From the repository root on the GPU server:

```bash
nohup .venv/bin/python -u scripts/run_model_suite.py \
  > logs/model_suite.nohup.log 2>&1 &
```

Follow progress without keeping the notebook browser open:

```bash
tail -f logs/model_suite.nohup.log
```

Resume after a disconnect or server interruption by running the same `nohup`
command again. Completed files in `results/audit.*.jsonl` are skipped.
Existing Mistral smoke/full filenames from the notebook workflow are also
recognized:

- `results/audit.mistral7b.limit1000.jsonl`
- `results/audit.mistral7b-instruct.full.jsonl`

The runner writes:

- `results/wcs_summary.model_suite.csv`: context-level coverage over 1,000
  target-word occurrences.
- `results/wcs_word_summary.model_suite.csv`: word-level coverage over 100
  unique target words, including whether each word is covered in at least one
  context.

Run a subset by slug:

```bash
nohup .venv/bin/python -u scripts/run_model_suite.py \
  --models mistral7b-v03-instruct,llama31-8b-base \
  > logs/model_suite.nohup.log 2>&1 &
```

The default suite uses verified Hugging Face IDs where available:

- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-v0.3`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen3.5-9B-Base`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen2.5-14B`
- `Qwen/Qwen2.5-14B-Instruct`
- `google/gemma-3-12b-pt`
- `google/gemma-3-12b-it`
- `google/gemma-4-E4B`
- `google/gemma-4-E4B-it`
- `google/gemma-2-9b`
- `google/gemma-2-9b-it`
- `deepseek-ai/DeepSeek-V2-Lite`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

For the DeepSeek pair, the runner uses `Qwen/Qwen2.5-14B` as the base
comparison for `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, matching the
distilled model's documented base family.

`deepseek-ai/DeepSeek-V2-Lite` requires `--trust-remote-code` when run through
Transformers.
