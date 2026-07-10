from __future__ import annotations

import unittest

import torch

from wcs.audit import (
    audit_sample,
    audit_sample_temperatures,
    completed_sample_ids,
    retain_complete_samples,
    patch_transformers_remote_code_compatibility,
    rank_probability_from_logits,
)
from pathlib import Path
from tempfile import TemporaryDirectory
import json


class FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self.param

    def __call__(self, input_ids: torch.Tensor) -> FakeOutput:
        vocab_size = 8
        logits = torch.full((1, input_ids.shape[1], vocab_size), -10.0)
        step = input_ids.shape[1]
        if step == 3:
            logits[0, -1, 3] = 4.0
            logits[0, -1, 4] = 3.0
            logits[0, -1, 5] = 2.0
        else:
            logits[0, -1, 4] = 4.0
            logits[0, -1, 3] = 3.0
            logits[0, -1, 5] = 2.0
        return FakeOutput(logits)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "alpha beta gamma":
            return [1, 2, 3]
        if text == " target":
            return [3, 4]
        raise AssertionError(f"unexpected text to encode: {text!r}")

    def decode(self, token_ids: list[int]) -> str:
        return {3: " tar", 4: "get", 5: " other"}.get(
            token_ids[0], f"<token-{token_ids[0]}>"
        )


class AuditTest(unittest.TestCase):
    def test_completed_sample_ids_ignores_partial_paths(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            rows = [
                {"sample_id": "done", "word_token_index": 0, "word_token_count": 2},
                {"sample_id": "done", "word_token_index": 1, "word_token_count": 2},
                {"sample_id": "partial", "word_token_index": 0, "word_token_count": 2},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(completed_sample_ids(path), {"done"})
            retain_complete_samples(path, {"done"})
            retained = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(retained), 2)
            self.assertEqual({row["sample_id"] for row in retained}, {"done"})

    def test_rank_probability_from_logits(self) -> None:
        logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
        result = rank_probability_from_logits(logits, token_id=3, neighbor_count=1)

        self.assertEqual(result.rank, 2)
        self.assertGreater(result.probability, 0)
        self.assertGreater(result.top_probability, result.probability)
        self.assertGreater(result.probability_ratio_to_top, 0)
        self.assertGreater(result.cumulative_probability, result.probability)
        self.assertEqual([token.token_id for token in result.top_tokens], [1, 3, 2, 0])
        self.assertEqual([token.token_id for token in result.neighbors_above], [1])
        self.assertEqual([token.token_id for token in result.neighbors_below], [2])

    def test_rank_probability_applies_temperature_scaling(self) -> None:
        logits = torch.tensor([3.0, 2.0, 0.0])
        cold = rank_probability_from_logits(
            logits, token_id=1, temperature=0.5
        )
        warm = rank_probability_from_logits(
            logits, token_id=1, temperature=1.5
        )

        self.assertGreater(warm.probability, cold.probability)
        self.assertGreater(cold.top_probability, warm.top_probability)

    def test_audit_sample_walks_target_tokens(self) -> None:
        sample = {
            "id": "sample-000001",
            "word": "target",
            "rank": 12345,
            "source_path": "fixture.txt",
            "prefix": "alpha beta gamma",
            "context_token_count": 3,
        }

        rows = audit_sample(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            sample=sample,
            model_name="fake-model",
            device="cpu",
            temperature=0.7,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].sample_id, "sample-000001")
        self.assertEqual(rows[0].token_id, 3)
        self.assertEqual(rows[0].rank, 1)
        self.assertEqual(rows[1].token_id, 4)
        self.assertEqual(rows[1].rank, 1)
        self.assertEqual(rows[0].word_token_count, 2)
        self.assertEqual(rows[0].temperature, 0.7)
        self.assertEqual(rows[0].audit_schema_version, 2)
        self.assertEqual(rows[0].rank_neighbor_count, 5)
        self.assertEqual(rows[0].top_5_tokens[:3], [" tar", "get", " other"])
        self.assertEqual(rows[0].rank_neighbors_above, [])
        self.assertEqual(rows[0].rank_neighbors_below[0].token_text, "get")

    def test_audit_sample_temperatures_reuses_path_for_multiple_temperatures(self) -> None:
        sample = {
            "id": "sample-000001",
            "word": "target",
            "rank": 12345,
            "source_path": "fixture.txt",
            "prefix": "alpha beta gamma",
            "context_token_count": 3,
        }

        rows_by_temperature = audit_sample_temperatures(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            sample=sample,
            model_name="fake-model",
            temperatures=[1.0, 0.7],
            device="cpu",
        )

        self.assertEqual(set(rows_by_temperature), {1.0, 0.7})
        self.assertEqual(len(rows_by_temperature[1.0]), 2)
        self.assertEqual(len(rows_by_temperature[0.7]), 2)
        self.assertEqual(rows_by_temperature[1.0][0].temperature, 1.0)
        self.assertEqual(rows_by_temperature[0.7][0].temperature, 0.7)
        self.assertEqual(rows_by_temperature[1.0][0].token_id, rows_by_temperature[0.7][0].token_id)

    def test_remote_code_compatibility_patch_adds_deprecated_helpers(self) -> None:
        import transformers.pytorch_utils as pytorch_utils
        import transformers.utils.import_utils as import_utils

        original_fx = getattr(import_utils, "is_torch_fx_available", None)
        original_torch_check = getattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13", None)
        if hasattr(import_utils, "is_torch_fx_available"):
            delattr(import_utils, "is_torch_fx_available")
        if hasattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13"):
            delattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13")
        try:
            patch_transformers_remote_code_compatibility()
            self.assertTrue(import_utils.is_torch_fx_available())
            self.assertTrue(pytorch_utils.is_torch_greater_or_equal_than_1_13)
        finally:
            if original_fx is not None:
                import_utils.is_torch_fx_available = original_fx
            elif hasattr(import_utils, "is_torch_fx_available"):
                delattr(import_utils, "is_torch_fx_available")
            if original_torch_check is not None:
                pytorch_utils.is_torch_greater_or_equal_than_1_13 = original_torch_check
            elif hasattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13"):
                delattr(pytorch_utils, "is_torch_greater_or_equal_than_1_13")


if __name__ == "__main__":
    unittest.main()
