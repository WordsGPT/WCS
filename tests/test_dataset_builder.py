from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from wcs.dataset_builder import (
    _parse_coherence_response,
    build_samples,
    index_corpus_occurrences,
    load_frequency_entries,
    normalize_target_word,
    write_jsonl,
)


FIXTURES = Path(__file__).parent / "fixtures"


class DatasetBuilderTest(unittest.TestCase):
    def test_load_frequency_entries_assigns_expected_fields(self) -> None:
        entries = load_frequency_entries(FIXTURES / "frequency" / "norvig_sample.tsv")

        self.assertEqual(entries[2].rank, 3)
        self.assertEqual(entries[2].word, "elaborate")
        self.assertEqual(entries[2].count, 50000)

    def test_build_samples_finds_full_contexts(self) -> None:
        with patch("wcs.dataset_builder.is_text_coherent", return_value=True):
            samples, missing = build_samples(
                frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
                corpus_path=FIXTURES / "pg19_sample",
                rank_min=3,
                rank_max=8,
                sample_size=4,
                context_tokens=5,
                seed=7,
                min_word_length=3,
                contexts_per_word=1,
                skip_coherence_check=True,
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

    def test_index_corpus_occurrences_preserves_prefix_punctuation(self) -> None:
        with TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "book.txt"
            corpus_path.write_text(
                'Alpha, beta; "gamma" delta epsilon target arrives.',
                encoding="utf-8",
            )

            occurrences, _ = index_corpus_occurrences(
                words={"target"},
                corpus_files=[corpus_path],
                context_tokens=5,
            )

        self.assertEqual(len(occurrences["target"]), 1)
        occurrence = occurrences["target"][0]
        self.assertEqual(
            occurrence.prefix,
            'Alpha, beta; "gamma" delta epsilon',
        )
        self.assertEqual(occurrence.context_token_count, 5)
        self.assertIn('"gamma"', occurrence.raw_excerpt)

    def test_unicode_words_are_normalized_and_indexed(self) -> None:
        self.assertEqual(normalize_target_word("después"), "después")
        self.assertEqual(normalize_target_word("doña"), "doña")

        with TemporaryDirectory() as directory:
            corpus_path = Path(directory) / "book.txt"
            corpus_path.write_text(
                "uno dos tres cuatro cinco Después llega doña Jacinta.",
                encoding="utf-8",
            )

            occurrences, _ = index_corpus_occurrences(
                words={"después", "doña"},
                corpus_files=[corpus_path],
                context_tokens=5,
            )

        self.assertEqual(len(occurrences["después"]), 1)
        self.assertEqual(occurrences["después"][0].matched_text, "Después")
        self.assertEqual(len(occurrences["doña"]), 1)

    def test_build_samples_can_exclude_capitalized_matches(self) -> None:
        with patch("wcs.dataset_builder.is_text_coherent", return_value=True):
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
                contexts_per_word=1,
                skip_coherence_check=True,
            )

        self.assertFalse(missing)
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(not sample.matched_text[:1].isupper() for sample in samples))

    def test_build_samples_can_filter_short_words(self) -> None:
        with patch("wcs.dataset_builder.is_text_coherent", return_value=True):
            samples, missing = build_samples(
                frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
                corpus_path=FIXTURES / "pg19_sample",
                rank_min=3,
                rank_max=8,
                sample_size=6,
                context_tokens=5,
                seed=7,
                min_word_length=8,
                contexts_per_word=1,
                skip_coherence_check=True,
            )

        self.assertFalse(missing)
        self.assertTrue(all(len(sample.word) >= 8 for sample in samples))

    def test_build_samples_can_skip_coherence_check(self) -> None:
        with patch("wcs.dataset_builder.validate_contexts_with_gemini") as coherent:
            samples, missing = build_samples(
                frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
                corpus_path=FIXTURES / "pg19_sample",
                rank_min=3,
                rank_max=8,
                sample_size=4,
                context_tokens=5,
                seed=7,
                min_word_length=3,
                contexts_per_word=1,
                skip_coherence_check=True,
            )

        coherent.assert_not_called()
        self.assertFalse(missing)
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(sample.metadata["skip_coherence_check"] == 1 for sample in samples))

    def test_build_samples_can_filter_by_dictionary(self) -> None:
        with patch("wcs.dataset_builder.is_text_coherent", return_value=True):
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
                contexts_per_word=1,
                skip_coherence_check=True,
            )

        self.assertFalse(missing)
        self.assertTrue(samples)
        self.assertNotIn("artifact", {sample.word for sample in samples})

    def test_build_samples_can_resume_complete_word_groups(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "samples.jsonl"
            with patch("wcs.dataset_builder.is_text_coherent", return_value=True):
                first_samples, _ = build_samples(
                    frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
                    corpus_path=FIXTURES / "pg19_sample",
                    rank_min=3,
                    rank_max=8,
                    sample_size=1,
                    context_tokens=5,
                    seed=7,
                    min_word_length=3,
                    contexts_per_word=1,
                    checkpoint_path=output,
                    skip_coherence_check=True,
                )

                resumed_samples, _ = build_samples(
                    frequency_path=FIXTURES / "frequency" / "norvig_sample.tsv",
                    corpus_path=FIXTURES / "pg19_sample",
                    rank_min=3,
                    rank_max=8,
                    sample_size=2,
                    context_tokens=5,
                    seed=7,
                    min_word_length=3,
                    contexts_per_word=1,
                    checkpoint_path=output,
                    resume=True,
                    skip_coherence_check=True,
                )

        self.assertEqual(len(first_samples), 1)
        self.assertEqual(len(resumed_samples), 2)
        self.assertEqual([sample.id for sample in resumed_samples], [
            "sample-000001",
            "sample-000002",
        ])
        self.assertEqual(len({sample.word for sample in resumed_samples}), 2)

    def test_build_samples_batches_candidate_contexts_per_word(self) -> None:
        def accept_first(texts: list[str], **_: object) -> list[bool]:
            return [index == 0 for index, _text in enumerate(texts)]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            frequency = root / "frequency.tsv"
            frequency.write_text("1\ttarget\t3\n", encoding="utf-8")
            corpus = root / "book.txt"
            corpus.write_text(
                "uno dos tres cuatro cinco target. "
                "seis siete ocho nueve diez target. "
                "once doce trece catorce quince target.",
                encoding="utf-8",
            )
            with (
                patch("wcs.dataset_builder._gemini_api_key", return_value="test-key"),
                patch(
                    "wcs.dataset_builder.validate_contexts_with_gemini",
                    side_effect=accept_first,
                ) as validate,
            ):
                samples, missing = build_samples(
                    frequency_path=frequency,
                    corpus_path=corpus,
                    rank_min=1,
                    rank_max=1,
                    sample_size=1,
                    context_tokens=5,
                    seed=7,
                    min_word_length=3,
                    contexts_per_word=1,
                    candidate_contexts_per_word=3,
                    coherence_workers=2,
                )

        self.assertFalse(missing)
        self.assertEqual(len(samples), 1)
        validate.assert_called_once()
        self.assertEqual(len(validate.call_args.args[0]), 3)

    def test_parse_coherence_response_requires_one_boolean_per_context(self) -> None:
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"accepted":[true,false,true]}'}]
                    }
                }
            ]
        }

        self.assertEqual(
            _parse_coherence_response(response, 3),
            [True, False, True],
        )
        with self.assertRaises(ValueError):
            _parse_coherence_response(response, 2)


if __name__ == "__main__":
    unittest.main()
