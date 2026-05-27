# Frequency vs. WCS Experiment

This experiment samples 1,000 target words across three frequency-rank strata,
runs forced-path WCS audits, and reports per-word WCS correlations with
frequency rank and log frequency count.

## Default sample

The default strata are:

| Rank band | Words |
| --- | ---: |
| 1-1,000 | 200 |
| 1,001-10,000 | 400 |
| 10,001-100,000 | 400 |

The bands are non-overlapping so each target word has one frequency-band label.
Each selected word keeps 10 contexts by default, so the full sample contains
10,000 audited contexts.

## Representative models

Using the existing temperature-1 word summary
`wcs_word_summary_punct_all_temperatures.csv`, the four-model default is:

| Runner slug | Model | Existing mean context WCS |
| --- | --- | ---: |
| `gemma2-9b-base` | `google/gemma-2-9b` | 0.026 |
| `gemma3-12b-it` | `google/gemma-3-12b-it` | 0.139 |
| `mistral7b-v03-base` | `mistralai/Mistral-7B-v0.3` | 0.312 |
| `llama31-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 0.350 |

These are the four medoids of the existing WCS curve vectors, so they cover the
observed low, mid-low, mid, and high-coverage regimes without running the full
17-model suite. If the goal is to include the strongest observed model instead
of a medoid, replace `llama31-8b-instruct` with `qwen25-14b-base`.

## Server run

From the repository root on the GPU server:

```bash
nohup env CORPUS=/path/to/pg19/test \
  bash scripts/run_frequency_wcs_experiment.sh \
  > logs/frequency_wcs.nohup.log 2>&1 &
```

The script writes:

| Path | Contents |
| --- | --- |
| `data/processed/samples.frequency_1k.jsonl` | Stratified 1,000-word sample set. |
| `results/frequency_wcs/audit.<model>.jsonl` | Forced-path audit rows. |
| `results/frequency_wcs/wcs_summary.csv` | Context-level WCS by decoder/parameter. |
| `results/frequency_wcs/wcs_word_summary.csv` | Word-level any/all WCS by decoder/parameter. |
| `results/frequency_wcs/per_word_wcs.csv` | Per-word mean WCS by model plus `ALL_MODELS`, including target-token counts. |
| `results/frequency_wcs/frequency_correlations.csv` | Pearson/Spearman correlations against rank, log count, and token count. |
| `results/frequency_wcs/plots/wcs_vs_avg_token_count.first_100_words.png` | Pooled WCS vs average token count across models for the first 100 sampled words. |
| `results/frequency_wcs/plots/wcs_vs_token_count.<model>.first_100_words.png` | Per-model WCS vs target-token count for the first 100 sampled words. |

The run is resumable. Existing complete audit files are skipped by
`scripts/run_model_suite.py`; sample-building checkpoints live next to the
sample output under `data/processed/samples.frequency_1k.checkpoints/`.

Useful overrides:

```bash
MODELS=gemma2-9b-base,gemma3-12b-it,mistral7b-v03-base,qwen25-14b-base \
DTYPE=float16 \
TOKEN_PLOT_WORD_LIMIT=100 \
CORPUS=/path/to/corpus \
scripts/run_frequency_wcs_experiment.sh
```

Set `TOKEN_PLOT_WORD_LIMIT=0` to plot all sampled target words after the first
100-word check looks reasonable.

## Existing 100-word Tokenization Check

For the already-built 100-word sample, use the standalone tokenization plotter
against existing audit JSONL files:

```bash
python scripts/plot_tokenization_wcs.py \
  --samples data/processed/samples.jsonl \
  --output-dir results/tokenization_wcs_100 \
  --word-limit 100 \
  results
```

This writes `tokenization_wcs.first_100_words.csv`, token-count correlations,
one pooled WCS-vs-average-token-count plot, and one per-model WCS-vs-token-count
plot for each audited model.
