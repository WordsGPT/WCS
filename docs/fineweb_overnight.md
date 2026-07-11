# FineWeb overnight WCS and lexical-diversity run

This workflow streams the official `HuggingFaceFW/fineweb` `sample-10BT`
configuration and caches only target-bearing excerpts. The official 10BT sample
is about 27.6 GB; streaming avoids downloading all of it before useful work can
start and makes the context cache resumable.

## What the default run does

- Selects 200 target words from Norvig ranks 10,000–40,000.
- Finds 80 candidate FineWeb contexts per word and asks Gemini to retain 50.
- Runs the six-model compact WCS suite at temperature 1.0.
- Summarizes top-k for k=1..20, then 25, 30, ..., 80. The same logits cover all
  k values, so this adds no GPU inference work.
- Performs unconstrained sampling (`top_k=0`, `top_p=1`, temperature 1.0) on a
  fixed random subset of 50 of those contexts per model.
- Keeps the first 200 generated word tokens and reports per-context mean/SD and
  pooled TTR and MTLD. MTLD uses the usual 0.72 threshold and the same forward
  algorithm exposed by the `LexicalRichness` package used in *Beware of Words*.

Generation is deliberately limited to 50 matched contexts per model. Generating
200 words for all 10,000 WCS contexts over six models is 12 million generated
words and is not realistic on one GPU in nine hours. Set `GEN_CONTEXTS=10000`
only for a later full run.

## Server commands

After pushing these files, on the server:

```bash
git pull
scripts/setup_overnight.sh
```

Do not walk away until setup and the preflight model smoke tests pass. Set the
credentials in the shell or in a repository-root `.env`:

```bash
export HF_TOKEN=...
export GEMINI_API_KEY=...
```

Start the resumable job:

```bash
nohup scripts/run_fineweb_overnight.sh \
  > logs/fineweb_launcher.log 2>&1 &
echo $! > logs/fineweb_launcher.pid
tail -f logs/fineweb_200x50/overnight.log
```

Re-running the same command is safe. FineWeb candidates, accepted word groups,
WCS samples within a model, completed models, and generated contexts are all
checkpointed. A failure prints the relevant per-model log path.

The bootstrap creates an isolated `.venv`, pins the Hugging Face stack, reuses a
working system CUDA PyTorch when available, runs all unit tests, and otherwise
installs the official cu128 PyTorch wheel. The preflight then verifies free disk,
47/48 GB-class GPU memory, FineWeb streaming, Gemini structured output, Git
remote access, Hugging Face gating, and one real bfloat16 CUDA forward pass for
each selected model.

## Useful overrides

Use a smaller first pass if FineWeb collection or Gemini quota is slower than
expected:

```bash
N_WORDS=100 CONTEXTS_PER_WORD=10 GEN_CONTEXTS=50 \
RUN_ID=fineweb_100x10 scripts/run_fineweb_overnight.sh
```

Use more parallel Gemini requests only if the API quota supports them:

```bash
COHERENCE_WORKERS=20 scripts/run_fineweb_overnight.sh
```

Batch size defaults to one to avoid model-specific OOM surprises on a 47 GB
card. The Python CLIs expose the remaining model and generation controls.

## Outputs

- `data/processed/samples.fineweb_200x50.jsonl`: accepted WCS contexts.
- `results/fineweb_200x50/wcs/wcs_summary.csv`: context-level WCS.
- `results/fineweb_200x50/wcs/wcs_word_summary.csv`: target-word WCS.
- `results/fineweb_200x50/generation/generation.*.jsonl`: generated text and
  per-context metrics.
- `results/fineweb_200x50/generation/lexical_diversity.csv`: model summary.
- `logs/fineweb_200x50/overnight.log`: top-level progress and failures.

## Matched downstream validation

After the 200x50 sample file exists, run the reviewer-requested downstream
experiment independently of corpus collection and the full WCS audit:

```bash
nohup scripts/run_downstream_validation.sh \
  > logs/fineweb_downstream_launcher.log 2>&1 &
```

The production defaults retain all six Base/Instruct models and use the same 50
FineWeb contexts for every condition. Base models receive raw continuation
prefixes; Instruct models receive their native chat template and an explicit
continuation instruction. The script audits and generates from the exact same
tokenized prompts, varies top-k (10, 20, 50, 80), top-p (0.80, 0.90, 0.95,
0.99), min-p (0.01, 0.05, 0.10), and an untruncated baseline at temperatures
0.7 and 1.0. Generation stops normally at EOS and diversity is scored on the
first 100 lexical words. Early-stop/completion rates are reported separately.

Outputs are resumable under `results/fineweb_200x50/downstream_validation/`:

- `conditioned_wcs/wcs_summary.csv`: WCS on the exact generation prompts.
- `generations/generation.*.jsonl`: per-context generations and metrics.
- `lexical_diversity_by_config.csv`: TTR/MTLD summaries and confidence intervals.
- `wcs_diversity_correlations.csv`: Pearson and Spearman correlations with
  two-sided permutation p-values.

FineWeb is web-crawl data and may contain residual toxic, biased, or personal
content despite its filtering. Treat the generated JSONL as research data.
