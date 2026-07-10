from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wcs.metrics import DEFAULT_TOP_K, summarize_wcs, summarize_wcs_by_target_word, write_summary_csv


def audit_row(
    sample_id: str,
    token_index: int,
    rank: int,
    cumulative_probability: float,
    probability_ratio_to_top: float,
    word: str | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "model": "fake-model",
        "word": word or sample_id,
        "word_rank": 12345,
        "source_path": "fixture.txt",
        "word_token_index": token_index,
        "token_id": token_index + 10,
        "token_text": "x",
        "rank": rank,
        "probability": 0.1,
        "top_probability": 0.2,
        "probability_ratio_to_top": probability_ratio_to_top,
        "cumulative_probability": cumulative_probability,
        "context_token_count": 256,
        "prefix_char_count": 1000,
        "word_token_count": 2,
    }


class MetricsTest(unittest.TestCase):
    def test_default_top_k_schedule(self) -> None:
        self.assertEqual(DEFAULT_TOP_K, tuple(range(1, 21)) + tuple(range(25, 81, 5)))

    def test_summarize_wcs_requires_all_word_tokens_to_survive(self) -> None:
        with TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.jsonl"
            rows = [
                audit_row("word-a", 0, rank=1, cumulative_probability=0.20, probability_ratio_to_top=0.9),
                audit_row("word-a", 1, rank=2, cumulative_probability=0.40, probability_ratio_to_top=0.8),
                audit_row("word-b", 0, rank=1, cumulative_probability=0.20, probability_ratio_to_top=0.9),
                audit_row("word-b", 1, rank=5, cumulative_probability=0.96, probability_ratio_to_top=0.01),
            ]
            with audit_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summaries = summarize_wcs(
                [audit_path],
                top_k_values=[2, 5],
                top_p_values=[0.50, 0.99],
                min_p_values=[0.05, 0.95],
            )
            keyed = {(row.decoder, row.parameter): row for row in summaries}

            self.assertEqual(keyed[("top_k", 2.0)].covered_words, 1)
            self.assertEqual(keyed[("top_k", 5.0)].covered_words, 2)
            self.assertEqual(keyed[("top_p", 0.5)].covered_words, 1)
            self.assertEqual(keyed[("top_p", 0.99)].covered_words, 2)
            self.assertEqual(keyed[("min_p", 0.05)].covered_words, 1)
            self.assertEqual(keyed[("min_p", 0.95)].covered_words, 0)

            summary_path = Path(directory) / "summary.csv"
            write_summary_csv(summaries, summary_path)
            self.assertIn("decoder", summary_path.read_text(encoding="utf-8"))

    def test_summarize_wcs_by_target_word_counts_any_surviving_context(self) -> None:
        with TemporaryDirectory() as directory:
            audit_path = Path(directory) / "audit.jsonl"
            rows = [
                audit_row("word-a-context-1", 0, rank=1, cumulative_probability=0.20, probability_ratio_to_top=0.9, word="word-a"),
                audit_row("word-a-context-2", 0, rank=50, cumulative_probability=0.99, probability_ratio_to_top=0.001, word="word-a"),
                audit_row("word-b-context-1", 0, rank=50, cumulative_probability=0.99, probability_ratio_to_top=0.001, word="word-b"),
                audit_row("word-b-context-2", 0, rank=60, cumulative_probability=0.99, probability_ratio_to_top=0.001, word="word-b"),
            ]
            with audit_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            summaries = summarize_wcs_by_target_word(
                [audit_path],
                top_k_values=[10],
                top_p_values=[0.50],
                min_p_values=[0.05],
            )
            keyed = {(row.decoder, row.parameter): row for row in summaries}

            top_k = keyed[("top_k", 10.0)]
            self.assertEqual(top_k.total_words, 2)
            self.assertEqual(top_k.total_contexts, 4)
            self.assertEqual(top_k.covered_words_any, 1)
            self.assertEqual(top_k.covered_words_all, 0)
            self.assertEqual(top_k.covered_contexts, 1)
            self.assertEqual(top_k.word_any_wcs, 0.5)
            self.assertEqual(top_k.word_all_wcs, 0.0)

            top_p = keyed[("top_p", 0.5)]
            self.assertEqual(top_p.covered_words_any, 1)
            self.assertEqual(top_p.covered_contexts, 1)

            min_p = keyed[("min_p", 0.05)]
            self.assertEqual(min_p.covered_words_any, 1)
            self.assertEqual(min_p.covered_contexts, 1)


if __name__ == "__main__":
    unittest.main()
