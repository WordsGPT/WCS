from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BuildExplorerDataTests(unittest.TestCase):
    def test_cli_builds_multi_token_word_and_first_token_neighbors(self) -> None:
        sample = {
            "id": "sample-000001",
            "word": "target",
            "rank": 123,
            "prefix": "A prefix ",
        }
        common = {
            "sample_id": "sample-000001",
            "model": "example/model",
            "word": "target",
            "temperature": 1.0,
            "audit_schema_version": 2,
            "rank_neighbor_count": 2,
            "rank_neighbors_above": [],
            "rank_neighbors_below": [],
        }
        first = {
            **common,
            "word_token_index": 0,
            "token_id": 10,
            "token_text": " tar",
            "rank": 3,
            "probability": 0.5,
            "top_5_tokens": [" one", " two", " tar", " four", " five"],
            "top_5_probs": [0.7, 0.6, 0.5, 0.4, 0.3],
            "rank_neighbors_above": [
                {"rank": 1, "token_id": 1, "token_text": " one", "probability": 0.7},
                {"rank": 2, "token_id": 2, "token_text": " two", "probability": 0.6},
            ],
            "rank_neighbors_below": [
                {"rank": 4, "token_id": 4, "token_text": " four", "probability": 0.4},
                {"rank": 5, "token_id": 5, "token_text": " five", "probability": 0.3},
            ],
        }
        second = {
            **common,
            "word_token_index": 1,
            "token_id": 11,
            "token_text": "get",
            "rank": 7,
            "probability": 0.25,
            "top_5_tokens": ["a", "b", "c", "d", "e"],
            "top_5_probs": [0.5, 0.4, 0.3, 0.2, 0.1],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            samples = temp / "samples.jsonl"
            audit = temp / "audit.jsonl"
            output = temp / "explorer.json"
            samples.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            audit.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_explorer_data.py"),
                    "--samples",
                    str(samples),
                    "--audits",
                    str(audit),
                    "--output",
                    str(output),
                    "--dataset",
                    "English PG-19",
                    "--language",
                    "English",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(output.read_text(encoding="utf-8"))

        result = data["words"]["target"]["contexts"][0]["results"]["example/model"]
        self.assertEqual(result["rank"], 7)
        self.assertEqual(result["prob"], 0.125)
        self.assertEqual(result["targetToken"]["rank"], 3)
        self.assertEqual([row["rank"] for row in result["neighborsAbove"]], [1, 2])
        self.assertEqual([row["rank"] for row in result["neighborsBelow"]], [4, 5])
        self.assertEqual([row["index"] for row in result["tokenSteps"]], [0, 1])
        self.assertEqual(data["metadata"]["dataset"], "English PG-19")


if __name__ == "__main__":
    unittest.main()
