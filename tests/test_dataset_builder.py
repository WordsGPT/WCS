from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wcs.dataset_builder import build_samples, load_frequency_entries, write_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


class DatasetBuilderTest(unittest.TestCase):
    def test_load_frequency_entries_assigns_expected_fields(self) -> None:
        entries = load_frequency_entries(FIXTURES / "frequency" / "norvig_sample.tsv")

        self.assertEqual(entries[2].rank, 3)
        self.assertEqual(entries[2].word, "elaborate")
        self.assertEqual(entries[2].count, 50000)

    def test_build_samples_finds_full_contexts(self) -> None:
        samples, missing = build_samples(
            frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
            corpus_path=FIXTURES / "pg19_sample",
            rank_min=3,
            rank_max=8,
            sample_size=4,
            context_tokens=5,
            seed=7,
            min_word_length=3,
        )

        self.assertFalse(missing)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample.context_token_count >= 5 for sample in samples))
        self.assertTrue(
            {sample.word for sample in samples}.issubset(
                {"elaborate", "eloquence", "indignant", "languor", "obdurate", "resplendent"}
            )
        )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "samples.jsonl"
            write_jsonl(samples, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertTrue(rows[0]["id"].startswith("sample-"))
        self.assertIn("prefix", rows[0])

    def test_build_samples_can_exclude_capitalized_matches(self) -> None:
        samples, missing = build_samples(
            frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
            corpus_path=FIXTURES / "pg19_sample",
            rank_min=3,
            rank_max=8,
            sample_size=6,
            context_tokens=5,
            seed=7,
            exclude_capitalized_matches=True,
            min_word_length=3,
        )

        self.assertFalse(missing)
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(not sample.matched_text[:1].isupper() for sample in samples))

    def test_build_samples_can_filter_short_words(self) -> None:
        samples, missing = build_samples(
            frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
            corpus_path=FIXTURES / "pg19_sample",
            rank_min=3,
            rank_max=8,
            sample_size=6,
            context_tokens=5,
            seed=7,
            min_word_length=8,
        )

        self.assertFalse(missing)
        self.assertTrue(all(len(sample.word) >= 8 for sample in samples))

    def test_build_samples_can_filter_by_dictionary(self) -> None:
        samples, missing = build_samples(
            frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
            corpus_path=FIXTURES / "pg19_sample",
            rank_min=3,
            rank_max=11,
            sample_size=6,
            context_tokens=5,
            seed=7,
            min_word_length=3,
            dictionary_path=FIXTURES / "frequency" / "dictionary_sample.txt",
        )

        self.assertFalse(missing)
        self.assertTrue(samples)
        self.assertNotIn("artifact", {sample.word for sample in samples})


if __name__ == "__main__":
    unittest.main()
