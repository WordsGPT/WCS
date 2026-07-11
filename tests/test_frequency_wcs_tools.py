import unittest

from scripts.build_stratified_frequency_samples import parse_strata
from scripts.summarize_frequency_wcs import (
    add_aggregate_rows,
    average_ranks,
    correlation_rows,
    first_n_word_rows,
    pearson,
    spearman,
    summarize_words,
)


class FrequencyWcsToolTests(unittest.TestCase):
    def test_parse_strata(self):
        self.assertEqual(
            parse_strata("1:1000:200,1001:10000:400"),
            [(1, 1000, 200), (1001, 10000, 400)],
        )

    def test_parse_strata_rejects_bad_ranges(self):
        with self.assertRaises(ValueError):
            parse_strata("1000:1:200")

    def test_average_ranks_handles_ties(self):
        self.assertEqual(average_ranks([10, 20, 20, 40]), [1.0, 2.5, 2.5, 4.0])

    def test_correlations(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(spearman([10, 20, 30], [3, 2, 1]), -1.0)

    def test_summarize_words_and_correlations(self):
        metadata = {
            "common": {
                "rank": 10,
                "count": 1000,
                "sample_contexts": 1,
                "frequency_band": "1-1000",
                "sample_order": 1,
            },
            "rare": {
                "rank": 20000,
                "count": 20,
                "sample_contexts": 1,
                "frequency_band": "10001-100000",
                "sample_order": 2,
            },
        }
        groups = {
            ("fixture", 1.0, "common", "sample-1"): [
                {
                    "word_token_index": 0,
                    "rank": 1,
                    "probability": 0.5,
                    "cumulative_probability": 0.5,
                    "probability_ratio_to_top": 1.0,
                    "word_token_count": 1,
                }
            ],
            ("fixture", 1.0, "rare", "sample-2"): [
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
        rows = summarize_words(metadata, groups, top_k=[10], top_p=[0.9], min_p=[0.01])
        self.assertEqual(len(rows), 2)
        by_word = {row["word"]: row for row in rows}
        self.assertEqual(by_word["common"]["mean_wcs"], 1.0)
        self.assertEqual(by_word["rare"]["mean_wcs"], 0.0)
        self.assertEqual(by_word["common"]["mean_token_count"], 1.0)
        self.assertEqual(by_word["rare"]["mean_token_count"], 3.0)

        correlations = correlation_rows(add_aggregate_rows(rows))
        aggregate = [row for row in correlations if row["model"] == "ALL_MODELS"][0]
        self.assertAlmostEqual(aggregate["rank_spearman"], -1.0)
        self.assertAlmostEqual(aggregate["log_count_spearman"], 1.0)
        self.assertAlmostEqual(aggregate["token_count_spearman"], -1.0)

    def test_first_n_word_rows_keeps_all_model_rows_for_selected_words(self):
        rows = [
            {"model": "m1", "word": "a", "sample_order": 1},
            {"model": "ALL_MODELS", "word": "a", "sample_order": 1},
            {"model": "m1", "word": "b", "sample_order": 2},
            {"model": "ALL_MODELS", "word": "b", "sample_order": 2},
            {"model": "m1", "word": "c", "sample_order": 3},
        ]
        kept = first_n_word_rows(rows, 2)
        self.assertEqual([row["word"] for row in kept], ["a", "a", "b", "b"])


if __name__ == "__main__":
    unittest.main()
