from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_model_suite import audit_is_complete, select_models


class ModelSuiteTests(unittest.TestCase):
    def test_english_pg19_preset_excludes_nemotron_and_deepseek_v2(self) -> None:
        models = select_models("english-pg19")
        slugs = {model.slug for model in models}
        self.assertEqual(len(models), 17)
        self.assertNotIn("deepseek-v2-lite", slugs)
        self.assertFalse(any("nemotron" in slug for slug in slugs))

    def test_resume_rejects_legacy_audit_without_prediction_schema(self) -> None:
        legacy = {"sample_id": "sample-1", "rank": 3}
        current = {
            "sample_id": "sample-1",
            "rank": 3,
            "audit_schema_version": 2,
            "top_5_tokens": ["a", "b", "c", "d", "e"],
            "top_5_probs": [0.5, 0.2, 0.1, 0.05, 0.01],
            "rank_neighbor_count": 5,
            "rank_neighbors_above": [],
            "rank_neighbors_below": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.jsonl"
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
            self.assertFalse(audit_is_complete(path, 1))
            path.write_text(json.dumps(current) + "\n", encoding="utf-8")
            self.assertTrue(audit_is_complete(path, 1))


if __name__ == "__main__":
    unittest.main()
