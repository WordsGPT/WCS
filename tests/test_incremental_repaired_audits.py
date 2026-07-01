import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.merge_repaired_audits import merge_audit_file
from scripts.prepare_repaired_sample_delta import write_delta


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class IncrementalRepairedAuditTests(unittest.TestCase):
    def test_delta_contains_only_changed_model_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.jsonl"
            repaired = root / "repaired.jsonl"
            output = root / "delta.jsonl"
            rows = [
                {
                    "id": "sample-1",
                    "word": "uno",
                    "prefix": "original",
                    "matched_text": "uno",
                    "source_path": "a",
                    "match_start_char": 1,
                    "match_end_char": 4,
                },
                {
                    "id": "sample-2",
                    "word": "dos",
                    "prefix": "same",
                    "matched_text": "dos",
                    "source_path": "b",
                    "match_start_char": 2,
                    "match_end_char": 5,
                },
            ]
            write_jsonl(original, rows)
            repaired_rows = [dict(row) for row in rows]
            repaired_rows[0]["prefix"] = "replacement"
            repaired_rows[1]["metadata"] = {"non_model_change": True}
            write_jsonl(repaired, repaired_rows)

            count = write_delta(original, repaired, output)
            delta = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual([row["id"] for row in delta], ["sample-1"])

    def test_merge_replaces_all_token_rows_for_changed_samples(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.jsonl"
            delta = root / "delta.jsonl"
            output = root / "output.jsonl"
            write_jsonl(
                base,
                [
                    {"sample_id": "sample-1", "word_token_index": 0, "rank": 10},
                    {"sample_id": "sample-1", "word_token_index": 1, "rank": 11},
                    {"sample_id": "sample-2", "word_token_index": 0, "rank": 20},
                ],
            )
            write_jsonl(
                delta,
                [{"sample_id": "sample-1", "word_token_index": 0, "rank": 1}],
            )

            merge_audit_file(base, delta, output, {"sample-1"})
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(
            [(row["sample_id"], row["rank"]) for row in rows],
            [("sample-1", 1), ("sample-2", 20)],
        )


if __name__ == "__main__":
    unittest.main()
