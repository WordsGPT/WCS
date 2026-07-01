import unittest

from scripts.repair_spanish_pd_contexts import (
    apply_replacements,
    audit_existing_samples,
    find_replacements,
)
from wcs.dataset_builder import ContextDecision, IndexedOccurrence, Sample


def sample(identifier: str, word: str, start: int, prefix: str) -> Sample:
    return Sample(
        id=identifier,
        word=word,
        rank=10_000,
        count=1,
        prefix=prefix,
        matched_text=word,
        source_path=f"data/processed/spanish_pd_books/contexts/book-{start}.txt",
        match_start_char=start,
        match_end_char=start + len(word),
        context_token_count=4,
        search_start_char=0,
        metadata={"language": "Spanish"},
    )


def occurrence(word: str, start: int, excerpt: str) -> IndexedOccurrence:
    return IndexedOccurrence(
        word=word,
        prefix=excerpt.removesuffix(word).strip(),
        raw_excerpt=excerpt,
        matched_text=word,
        source_path=f"data/processed/spanish_pd_books/contexts/book-{start}.txt",
        match_start_char=start,
        match_end_char=start + len(word),
        context_token_count=4,
        global_start_char=start * 10,
    )


class RepairSpanishPdContextsTests(unittest.TestCase):
    def test_replaces_only_rejected_rows_and_preserves_ids(self) -> None:
        rows = [
            sample("sample-1", "felino", 1, "un paso"),
            sample("sample-2", "felino", 2, "texto roto"),
        ]

        def audit_validator(texts, **_kwargs):
            if len(texts) == 2:
                return [
                    ContextDecision(True, "accepted", "Natural prose."),
                    ContextDecision(False, "ocr_corruption", "OCR substitution."),
                ]
            return [ContextDecision(True, "accepted", "Clean replacement.")]

        decisions, _ = audit_existing_samples(
            rows, validator=audit_validator, workers=1
        )
        replacements, logs, shortages = find_replacements(
            rows,
            decisions,
            {"felino": [occurrence("felino", 1, "used felino"),
                        occurrence("felino", 3, "andar felino")]},
            validator=audit_validator,
            workers=1,
        )
        repaired = apply_replacements(rows, decisions, replacements)

        self.assertFalse(shortages)
        self.assertEqual([row.id for row in repaired], ["sample-1", "sample-2"])
        self.assertEqual(repaired[0], rows[0])
        self.assertEqual(repaired[1].match_start_char, 3)
        self.assertEqual(repaired[1].metadata["quality_repaired"], 1)
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0]["selected_as_replacement"])

    def test_reports_shortage_without_reusing_existing_contexts(self) -> None:
        rows = [sample("sample-1", "felino", 1, "texto roto")]
        decisions = {
            "sample-1": ContextDecision(False, "ocr_corruption", "Broken scan.")
        }
        replacements, logs, shortages = find_replacements(
            rows,
            decisions,
            {"felino": [occurrence("felino", 1, "same felino")]},
            validator=lambda texts, **kwargs: [],
            workers=1,
        )

        self.assertEqual(replacements, {})
        self.assertEqual(logs, [])
        self.assertEqual(shortages, {"felino": (1, 0)})

    def test_replacement_candidates_are_validated_in_small_batches(self) -> None:
        rows = [sample("sample-1", "felino", 1, "texto roto")]
        decisions = {
            "sample-1": ContextDecision(False, "ocr_corruption", "Broken scan.")
        }
        batch_lengths = []

        def validator(texts, **_kwargs):
            batch_lengths.append(len(texts))
            return [
                ContextDecision(True, "accepted", "Clean prose.") for _ in texts
            ]

        find_replacements(
            rows,
            decisions,
            {
                "felino": [
                    occurrence("felino", start, f"andar {start} felino")
                    for start in range(2, 11)
                ]
            },
            validator=validator,
            workers=1,
            candidate_batch_size=4,
        )

        self.assertEqual(batch_lengths, [4, 4, 1])


if __name__ == "__main__":
    unittest.main()
