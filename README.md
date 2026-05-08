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

`--sample-size` is the number of target words. `--contexts-per-word` is the number of accepted text contexts kept for each word. The builder first filters out words without enough raw corpus contexts, then runs the required Gemini coherence check.

Use `--resume` to continue appending to an existing output file. Resume preserves complete word groups and continues sample IDs from the checkpoint.

Use `--exclude-capitalized-matches` for the paper dataset to reduce proper names and place names such as capitalized mythological, personal, or geographic references.

Use `--min-word-length 3` or higher to remove short frequency-list artifacts such as two-letter abbreviations.

Use `--dictionary /usr/share/dict/linux.words` to keep only targets found in the local English word list.

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
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B`
- `Qwen/Qwen2.5-14B-Instruct`
- `google/gemma-3-12b-pt`
- `google/gemma-3-12b-it`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`

The paper draft includes `Qwen3.5-9B` and `Gemma-4-E4B` placeholders. Those
names are not resolved to stable Hugging Face IDs in the runner; update
`DEFAULT_MODELS` in [scripts/run_model_suite.py](scripts/run_model_suite.py)
if the final model IDs differ.
