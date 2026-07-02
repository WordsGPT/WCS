# English PG-19 Context Explorer Run

The English explorer uses the existing paper dataset:

- `data/processed/samples.jsonl`
- 100 target words
- 10 PG-19 contexts per word
- 1,000 contexts total

The audit saves, for every forced target token:

- target token probability and rank
- global top five next-token predictions
- five predictions immediately above the target rank
- five predictions immediately below the target rank

For a multi-token target word, the explorer shows the neighborhood for the
first target token. It also reports the forced-path probability product and
worst token rank for the complete word.

## Server command

From the repository root:

```bash
git pull
bash scripts/run_english_pg19_explorer.sh start
```

The command creates `.venv-english-pg19`, installs a CUDA-compatible PyTorch
build and the pinned model stack, verifies the GPU, validates the PG-19 sample
shape, checks access/config/tokenizers for all 15 A100-compatible models, and
starts the resumable run in the background. Gemma 3 27B base/instruct and
Nemotron are excluded.

Monitor it with:

```bash
bash scripts/run_english_pg19_explorer.sh status
bash scripts/run_english_pg19_explorer.sh logs
```

If the server restarts or the process stops, run `start` again. Completed model
audits are validated for the current prediction schema and skipped.

At successful completion the worker writes:

```text
results/english_pg19_predictions/
logs/english_pg19_predictions/
explorer_data.english.json
```

Open `context_explorer.html?dataset=english` through the same static web server
used for the Spanish explorer.

## One-time access requirement

The Llama and Gemma repositories are gated. The launcher starts `hf auth login`
when no token is configured, but the account must already have accepted the
licenses on the relevant Hugging Face model pages. Preflight checks every model
before the background job starts, so access problems fail immediately.

## Smoke test

Use a separate smoke-test output directory automatically:

```bash
LIMIT=5 bash scripts/run_english_pg19_explorer.sh run
```

## Configuration overrides

The defaults reproduce the A100-compatible English paper suite without
Nemotron or the two Gemma 3 27B checkpoints:

```text
MODELS=english-pg19-a100
TOP_K=5
RANK_NEIGHBORS=5
DTYPE=bfloat16
DEVICE_MAP=auto
```

Override detected GPU memory only when necessary:

```bash
MAX_MEMORY=0=44GiB,1=44GiB,cpu=160GiB \
  bash scripts/run_english_pg19_explorer.sh start
```
