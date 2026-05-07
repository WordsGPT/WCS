# Notebook Usage

Use this flow on the cluster from a GPU-backed Jupyter notebook.

## Setup

From the repository root:

```python
import sys
from pathlib import Path

ROOT = Path("/home/Shodan/WCS")
sys.path.insert(0, str(ROOT / "src"))
```

## Load Samples

```python
from wcs.audit import load_samples

samples = load_samples(ROOT / "data/processed/samples.jsonl", limit=5)
samples[0]
```

## Load One Model

```python
from wcs.audit import load_hf_model_and_tokenizer

model_name = "gpt2"  # replace with the cluster model id/path
model, tokenizer = load_hf_model_and_tokenizer(
    model_name,
    device="cuda",
    dtype="float16",
)
```

For larger models, prefer a local checkpoint path on shared storage if the cluster does not allow Hugging Face downloads from compute nodes.

## Audit a Few Samples

```python
from wcs.audit import audit_sample

rows = audit_sample(model, tokenizer, samples[0], model_name=model_name, device="cuda")
for row in rows:
    print(row)
```

## Write JSONL Output

```python
from wcs.audit import audit_samples, write_audit_jsonl

rows = audit_samples(
    model=model,
    tokenizer=tokenizer,
    samples=samples,
    model_name=model_name,
    device="cuda",
)
write_audit_jsonl(rows, ROOT / "results/audit.gpt2.smoke.jsonl")
```

## CLI Equivalent

```bash
python scripts/run_audit.py \
  --samples data/processed/samples.jsonl \
  --output results/audit.gpt2.smoke.jsonl \
  --model gpt2 \
  --device cuda \
  --dtype float16 \
  --limit 5
```

## Output Fields

Each JSONL row is one target-word token in the forced path.

Important fields:

- `sample_id`: input sample id.
- `word`: target word.
- `word_token_index`: position inside the tokenized target word.
- `token_id` and `token_text`: model-tokenizer target token.
- `rank`: rank of the target token at that forced step.
- `probability`: scalar probability assigned to the target token.
- `top_probability`: probability of the most likely next token.
- `probability_ratio_to_top`: needed for Min-p survival.
- `cumulative_probability`: nucleus mass needed to include this token.

## Aggregate WCS

```python
from wcs.metrics import summarize_wcs, write_summary_csv

summaries = summarize_wcs([ROOT / "results/audit.gpt2.smoke.jsonl"])
write_summary_csv(summaries, ROOT / "results/wcs_summary.gpt2.smoke.csv")
```

If `matplotlib` is installed:

```python
from wcs.metrics import plot_summary

plot_summary(
    ROOT / "results/wcs_summary.gpt2.smoke.csv",
    ROOT / "plots",
)
```
