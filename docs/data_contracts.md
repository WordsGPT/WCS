# Data Contracts

These contracts define the files passed between the WCS pipeline stages.

## Frequency List Input

The dataset builder accepts a local text file with one word per row. Supported row formats:

```text
word
rank word
word count
rank,word,count
word,count
```

Rows may be comma-separated, tab-separated, or whitespace-separated. If an explicit rank is not present, rank is assigned by row order.

Required fields after parsing:

| Field | Type | Description |
| --- | --- | --- |
| `rank` | integer | Frequency rank, where lower means more frequent. |
| `word` | string | Target lexical type. |
| `count` | integer or null | Optional frequency count. |

## Sample Set Output

The dataset builder writes JSON Lines. Each row is one target word mapped to one naturalistic corpus context.

Default path:

```text
data/processed/samples.jsonl
```

Schema:

```json
{
  "id": "sample-000001",
  "word": "eloquence",
  "rank": 18231,
  "count": 41872,
  "prefix": "The speech was remembered for its unusual",
  "matched_text": "eloquence",
  "source_path": "pg19/test/book.txt",
  "match_start_char": 12345,
  "match_end_char": 12354,
  "context_token_count": 256,
  "search_start_char": 9123,
  "metadata": {
    "rank_min": 10000,
    "rank_max": 40000,
    "seed": 13
  }
}
```

Important details:

- `prefix` is the preceding context only. It does not include the target word.
- `matched_text` preserves the corpus spelling/casing that matched `word`.
- `context_token_count` is based on whitespace tokenization during dataset construction.
- `selection` is `filled_from_rank_band`: the builder randomly orders the requested frequency band, skips words that do not occur with full context, and continues until it reaches the requested sample count or exhausts the band.
- `exclude_capitalized_matches` records whether capitalized corpus occurrences were skipped to reduce proper names and place names.
- `min_word_length` records the minimum normalized frequency-list word length used to reduce short abbreviation/noise artifacts.
- `dictionary` records the optional dictionary file used to validate target words.
- Later model-audit code should re-tokenize `prefix` and `word` with each model tokenizer.

## Future Audit Output

The forced-path audit phase should consume `samples.jsonl` and emit token-level rows. This is intentionally not implemented yet.

Recommended future path:

```text
results/audit.<model_slug>.jsonl
```

Recommended future fields:

```json
{
  "sample_id": "sample-000001",
  "model": "meta-llama/Llama-3.1-8B",
  "word": "eloquence",
  "word_token_index": 0,
  "token_id": 1234,
  "token_text": " elo",
  "rank": 87,
  "prob": 0.00041,
  "cumprob_before_or_at_token": 0.934,
  "top_probability": 0.12
}
```
