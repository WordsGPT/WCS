from __future__ import annotations

import unittest

from scripts.run_downstream_validation import (
    Condition,
    average_ranks,
    permutation_correlation,
    prompt_for_sample,
)
from scripts.run_model_suite import ModelSpec


class MappingChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize:
            return {"input_ids": [[11, 12, 13]]}
        return "<chat>prompt"


class DownstreamValidationTest(unittest.TestCase):
    def test_conditions_enable_only_the_requested_filter(self) -> None:
        self.assertEqual(
            Condition("top_k", 50).generation_kwargs(),
            {"top_k": 50, "top_p": 1.0, "min_p": 0.0},
        )
        self.assertEqual(
            Condition("top_p", 0.95).generation_kwargs(),
            {"top_k": 0, "top_p": 0.95, "min_p": 0.0},
        )
        self.assertEqual(
            Condition("min_p", 0.05).generation_kwargs(),
            {"top_k": 0, "top_p": 1.0, "min_p": 0.05},
        )

    def test_average_ranks_and_permutation_correlation(self) -> None:
        self.assertEqual(average_ranks([10, 20, 20, 40]), [1.0, 2.5, 2.5, 4.0])
        statistic, p_value = permutation_correlation(
            [1, 2, 3, 4], [2, 4, 6, 8], ranked=False, permutations=100
        )
        self.assertAlmostEqual(statistic, 1.0)
        self.assertGreater(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_chat_template_accepts_mapping_input_ids(self) -> None:
        model = ModelSpec("fixture-it", "fixture/model", "fixture", "instruct")
        prompt, token_ids, mode = prompt_for_sample(
            MappingChatTokenizer(),
            model,
            {"prefix": "A passage"},
            requested_words=100,
        )
        self.assertEqual(prompt, "<chat>prompt")
        self.assertEqual(token_ids, [11, 12, 13])
        self.assertEqual(mode, "chat_continuation_instruction")


if __name__ == "__main__":
    unittest.main()
