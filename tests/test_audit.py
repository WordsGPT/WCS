from __future__ import annotations

import unittest

import torch

from wcs.audit import audit_sample, audit_sample_temperatures, rank_probability_from_logits


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
        return {3: " tar", 4: "get"}[token_ids[0]]


class AuditTest(unittest.TestCase):
    def test_rank_probability_from_logits(self) -> None:
        logits = torch.tensor([0.0, 3.0, 1.0, 2.0])
        rank, probability, top_probability, ratio, cumulative = rank_probability_from_logits(
            logits, token_id=3
        )

        self.assertEqual(rank, 2)
        self.assertGreater(probability, 0)
        self.assertGreater(top_probability, probability)
        self.assertGreater(ratio, 0)
        self.assertGreater(cumulative, probability)

    def test_rank_probability_applies_temperature_scaling(self) -> None:
        logits = torch.tensor([3.0, 2.0, 0.0])
        _rank, cold_probability, cold_top, _ratio, _cumulative = rank_probability_from_logits(
            logits, token_id=1, temperature=0.5
        )
        _rank, warm_probability, warm_top, _ratio, _cumulative = rank_probability_from_logits(
            logits, token_id=1, temperature=1.5
        )

        self.assertGreater(warm_probability, cold_probability)
        self.assertGreater(cold_top, warm_top)

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


if __name__ == "__main__":
    unittest.main()
