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
