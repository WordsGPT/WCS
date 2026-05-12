# WCS Agent Handoff

This repo is running Word Coverage Score experiments. The basic question is:
given a real book prefix and a hidden target word, would the target word survive
common decoding filters such as top-k, top-p, and min-p?

## Current Dataset Shape

The main paper-style dataset is:

- `data/processed/samples.jsonl`
- 100 target words
- 10 contexts per word
- 1000 samples total
- each sample stores 256 whitespace-delimited context tokens before the target word
- prefixes preserve punctuation and line breaks

The Moby-Dick dataset is:

- `data/processed/samples.mobydick.100x5.jsonl`
- built from root `mobydick.txt`
- 100 target words
- 5 contexts per word
- 500 samples total
- same 256 context-token prefix length

The Fortunata y Jacinta Spanish dataset should mirror the Moby-Dick shape:

- `data/processed/samples.fortunayjacinta.100x5.jsonl`
- built from root `FortunayJacinta.txt`
- 100 target words
- 5 contexts per word
- 256 context-token prefix length
- Spanish coherence prompt via `--language Spanish`

Important: `context_token_count: 256` is the dataset builder's word/context count,
not the evaluated model tokenizer's subword count.

## Core Scripts

Build samples:

```bash
python scripts/build_samples.py \
  --frequency data/raw/norvig_count_1w.txt \
  --corpus mobydick.txt \
  --output data/processed/samples.mobydick.100x5.jsonl \
  --rank-min 10000 \
  --rank-max 40000 \
  --sample-size 100 \
  --contexts-per-word 5 \
  --context-tokens 256 \
  --seed 13 \
  --min-word-length 3 \
  --dictionary /usr/share/dict/linux.words \
  --exclude-capitalized-matches \
  --progress-interval 1
```

This requires `GEMINI_API_KEY` for the coherence check.

Build singer-level lyric frequency lists:

```bash
python scripts/build_word_frequency.py \
  lyrics \
  --output-dir data/processed/wordlists/lyrics \
  --combined-name all_lyrics \
  --min-word-length 3
```

Build the Spanish frequency list from Fortunata y Jacinta:

```bash
python scripts/build_word_frequency.py \
  FortunayJacinta.txt \
  --group-name fortunayjacinta \
  --output-dir data/processed/wordlists \
  --min-word-length 4 \
  --strip-gutenberg
```

Build the external Spanish frequency list, matching the English Norvig-list
setup more closely:

```bash
python scripts/download_leipzig_frequency.py \
  --corpus spa_news_2023_1M \
  --output data/raw/spanish_frequency.tsv
```

Build Fortunata y Jacinta samples:

```bash
python scripts/build_samples.py \
  --frequency data/raw/spanish_frequency.tsv \
  --corpus FortunayJacinta.txt \
  --output data/processed/samples.fortunayjacinta.100x5.jsonl \
  --rank-min 1000 \
  --rank-max 12000 \
  --sample-size 100 \
  --contexts-per-word 5 \
  --context-tokens 256 \
  --seed 13 \
  --min-word-length 4 \
  --language Spanish \
  --progress-interval 1
```

Run model suite:

```bash
nohup .venv/bin/python -u scripts/run_model_suite.py \
  --samples data/processed/samples.mobydick.100x5.jsonl \
  --models llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,gemma3-12b-base,gemma3-12b-it \
  --temperatures 1.0,0.7,1.5 \
  --results-dir results/mobydick_100x5_multi_t \
  --logs-dir logs/mobydick_100x5_multi_t \
  --trust-remote-code \
  > logs/mobydick_100x5_multi_t.nohup.log 2>&1 &
```

Fortunata y Jacinta with the same six models/temperatures:

```bash
nohup .venv/bin/python -u scripts/run_model_suite.py \
  --samples data/processed/samples.fortunayjacinta.100x5.jsonl \
  --models llama31-8b-base,llama31-8b-instruct,mistral7b-v03-base,mistral7b-v03-instruct,gemma3-12b-base,gemma3-12b-it \
  --temperatures 1.0,0.7,1.5 \
  --results-dir results/fortunayjacinta_100x5_multi_t \
  --logs-dir logs/fortunayjacinta_100x5_multi_t \
  --trust-remote-code \
  > logs/fortunayjacinta_100x5_multi_t.nohup.log 2>&1 &
```

Monitor:

```bash
tail -f logs/mobydick_100x5_multi_t.nohup.log
```

## Models We Have Been Using

For the compact 6-model comparison:

- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-v0.3`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-3-12b-pt`
- `google/gemma-3-12b-it`

Runner slugs:

- `llama31-8b-base`
- `llama31-8b-instruct`
- `mistral7b-v03-base`
- `mistral7b-v03-instruct`
- `gemma3-12b-base`
- `gemma3-12b-it`

The broader runner also knows Qwen, Gemma 2, Gemma 4, DeepSeek-R1-Distill-Qwen,
and DeepSeek-V2-Lite.

## Temperature Handling

The audit can run multiple temperatures in one model pass:

```bash
--temperatures 1.0,0.7,1.5
```

This works because the model forward pass produces logits once, then the code
applies `softmax(logits / T)` for each requested temperature.

Important limitation: the audit JSONL files do not store raw logits or full
vocabulary distributions. They store only derived fields:

- `rank`
- `probability`
- `top_probability`
- `probability_ratio_to_top`
- `cumulative_probability`
- `temperature`

Therefore, new exact temperatures such as `0.8` or `1.2` require another audit
run. Existing outputs can only summarize the temperatures already run.

Top-k rank is temperature-invariant. Top-p and min-p are temperature-dependent.

## Result Files

Multi-temperature runs write per-temperature folders:

```text
results/<run_name>/t1/
results/<run_name>/t0p7/
results/<run_name>/t1p5/
```

Each temperature folder contains:

- `audit.<model_slug>.jsonl`
- `wcs_summary.csv`
- `wcs_word_summary.csv`

The word-level summary is usually the one used for graphs because our supervisor
cares whether a target word is ever covered across its contexts.

Relevant metrics:

- `word_any_wcs`: fraction of target words covered in at least one context
- `word_all_wcs`: fraction of target words covered in all contexts
- `covered_contexts / total_contexts`: context-level coverage

For finished Moby-Dick runs, combine per-temperature word summaries into one CSV
named `mobydick.csv` with a `temperature` column. The current site expects that
file at repo root.

## Existing Sites

Root graph pages:

- `index.html`: original decoder curves from `useforgraphs.csv`
- `temperature.html`: all-temperature paper-style comparison from `wcs_word_summary_punct_all_temperatures.csv`
- `mobydick.html`: Moby-Dick comparison from `mobydick.csv`

The HTML pages are static and fetch CSVs from the same directory. If opened via
`file://`, browser fetch may fail; use a simple local server:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/mobydick.html
```

## Server Notes

Server repo path has been:

```bash
/home/jovyan/WCS/WCS
```

Before running new jobs:

```bash
cd /home/jovyan/WCS/WCS
git pull
```

For Jupyter notebooks, backgrounding with `!cmd &` may fail. Use a real terminal
and `nohup ... &`.

Model files download into Hugging Face cache, generally under:

```bash
~/.cache/huggingface
```

## Spanish Gutenberg Corpus Export

Use the checked-in exporter rather than ad hoc Python heredocs. It inspects the
Hugging Face dataset configs and language values before exporting, and exits
with diagnostics if it picked a preview/config slice with no Spanish books.

Install the dependency once on the server if needed:

```bash
.venv/bin/pip install datasets
```

Inspect available configs/languages without writing files:

```bash
.venv/bin/python scripts/export_spanish_gutenberg.py --inspect-only
```

Export long Spanish books:

```bash
.venv/bin/python scripts/export_spanish_gutenberg.py \
  --output-dir data/raw/spanish_gutenberg \
  --min-chars 100000
```

If the default config candidates fail, rerun with a specific config shown by
`--inspect-only`:

```bash
.venv/bin/python scripts/export_spanish_gutenberg.py \
  --configs CONFIG_NAME \
  --output-dir data/raw/spanish_gutenberg \
  --min-chars 100000
```

## Known Compatibility Patches

`src/wcs/audit.py` includes compatibility shims for some Hugging Face remote
code under newer Transformers versions:

- missing `is_torch_fx_available`
- missing `DynamicCache.from_legacy_cache`
- model calls use `use_cache=False`

DeepSeek-V2-Lite is the most fragile model. Gemma 3 and Gemma 4 needed newer
Transformers than the first environment had.

## Git Notes

This workspace uses an unusual git setup:

```bash
git --git-dir=.git-real --work-tree=. status
git --git-dir=.git-real --work-tree=. add <files>
git --git-dir=.git-real --work-tree=. commit -m "message"
git --git-dir=.git-real --work-tree=. push
```

The remote prints a moved-repository notice but pushes have worked.

Ignored generated JSONL files sometimes need force-add:

```bash
git --git-dir=.git-real --work-tree=. add -f data/processed/samples.mobydick.100x5.jsonl
```

## Current Caveats

- We cannot derive new temperature values from old summaries or audits because
  logits are not stored.
- `mobydick.csv` is a combined summary CSV, not raw audit output.
- The raw book file `mobydick.txt` is local/untracked unless explicitly added.
- Keep base/instruct pairs together when possible; recent focus has been Llama,
  Mistral, and Gemma 3 pairs.
