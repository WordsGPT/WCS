from __future__ import annotations

import unittest

from wcs.lexical_diversity import lexical_tokens, mtld, type_token_ratio


class LexicalDiversityTest(unittest.TestCase):
    def test_tokenization_and_ttr_are_case_insensitive(self) -> None:
        tokens = lexical_tokens("Word word can't well-known 123")
        self.assertEqual(tokens, ["word", "word", "can't", "well-known"])
        self.assertEqual(type_token_ratio(tokens), 0.75)

    def test_mtld_rewards_nonrepeating_text(self) -> None:
        diverse = [f"word{i}" for i in range(100)]
        repetitive = ["same"] * 100
        self.assertGreater(mtld(diverse), mtld(repetitive))

    def test_mtld_validates_threshold_and_empty_input(self) -> None:
        self.assertEqual(mtld([]), 0.0)
        with self.assertRaises(ValueError):
            mtld(["word"], threshold=1.0)


if __name__ == "__main__":
    unittest.main()
