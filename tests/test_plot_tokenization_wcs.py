import unittest

from scripts.plot_tokenization_wcs import (
    add_average_token_rows,
    correlation_rows,
    selected_words,
    summarize_tokenization,
)


class PlotTokenizationWcsTests(unittest.TestCase):
    def test_selected_words_uses_sample_order_limit(self):
        metadata = {
            "third": {"sample_order": 3},
            "first": {"sample_order": 1},
            "second": {"sample_order": 2},
        }
        self.assertEqual(selected_words(metadata, 2), {"first", "second"})
        self.assertEqual(selected_words(metadata, 0), {"first", "second", "third"})

    def test_summarize_tokenization_and_aggregate_correlation(self):
        metadata = {
            "easy": {"sample_order": 1, "rank": 10, "count": 1000, "sample_contexts": 1},
            "hard": {"sample_order": 2, "rank": 20, "count": 500, "sample_contexts": 1},
        }
        groups = {
            ("model-a", 1.0, "easy", "sample-1"): [
                {
                    "word_token_index": 0,
                    "rank": 1,
                    "probability": 0.5,
                    "cumulative_probability": 0.5,
                    "probability_ratio_to_top": 1.0,
                    "word_token_count": 1,
                }
            ],
            ("model-a", 1.0, "hard", "sample-2"): [
                {
                    "word_token_index": 0,
                    "rank": 100,
                    "probability": 0.01,
                    "cumulative_probability": 0.99,
                    "probability_ratio_to_top": 0.001,
                    "word_token_count": 3,
                }
            ],
        }
        rows = summarize_tokenization(metadata, groups, top_k=[10], top_p=[0.9], min_p=[0.01])
        by_word = {row["word"]: row for row in rows}
        self.assertEqual(by_word["easy"]["mean_wcs"], 1.0)
        self.assertEqual(by_word["hard"]["mean_wcs"], 0.0)
        self.assertEqual(by_word["easy"]["mean_token_count"], 1.0)
        self.assertEqual(by_word["hard"]["mean_token_count"], 3.0)

        aggregate_rows = add_average_token_rows(rows)
        aggregate = [row for row in aggregate_rows if row["model"] == "ALL_MODELS"]
        self.assertEqual(len(aggregate), 2)
        aggregate_correlation = [
            row for row in correlation_rows(aggregate_rows) if row["model"] == "ALL_MODELS"
        ][0]
        self.assertAlmostEqual(aggregate_correlation["token_count_spearman"], -1.0)


if __name__ == "__main__":
    unittest.main()
