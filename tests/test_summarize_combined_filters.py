import unittest

from scripts.summarize_combined_filters import Recipe, parse_recipe, survives


class CombinedFilterSummaryTests(unittest.TestCase):
    def test_parse_recipe(self) -> None:
        self.assertEqual(
            parse_recipe("gemma:64:0.95"),
            Recipe(name="gemma", top_k=64, top_p=0.95),
        )

    def test_combined_filter_requires_both_conditions(self) -> None:
        recipe = Recipe(name="combined", top_k=3, top_p=0.95)
        self.assertTrue(
            survives(
                [
                    {
                        "rank": 3,
                        "cumulative_probability": 0.95,
                        "probability": 0.05,
                        "top_5_probs": [0.7, 0.2, 0.05],
                    }
                ],
                recipe,
            )
        )
        self.assertFalse(
            survives(
                [
                    {
                        "rank": 4,
                        "cumulative_probability": 0.96,
                        "probability": 0.01,
                        "top_5_probs": [0.7, 0.25, 0.01],
                    }
                ],
                recipe,
            )
        )
        self.assertFalse(
            survives(
                [
                    {
                        "rank": 3,
                        "cumulative_probability": 0.97,
                        "probability": 0.01,
                        "top_5_probs": [0.7, 0.25, 0.01],
                    }
                ],
                recipe,
            )
        )

    def test_top_p_is_renormalized_after_top_k(self) -> None:
        recipe = Recipe(name="combined", top_k=2, top_p=0.8)
        row = {
            "rank": 2,
            "cumulative_probability": 0.7,
            "probability": 0.1,
            "top_5_probs": [0.6, 0.1],
        }
        self.assertFalse(survives([row], recipe))


if __name__ == "__main__":
    unittest.main()
