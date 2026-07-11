from __future__ import annotations

import unittest

from scripts.analyze_downstream_validation import holm_adjust


class AnalyzeDownstreamValidationTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_in_sorted_p_values(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.20])
        self.assertEqual(adjusted, [0.04, 0.09, 0.09, 0.20])

    def test_holm_adjustment_preserves_input_order_and_caps_at_one(self) -> None:
        self.assertEqual(holm_adjust([0.9, 0.01]), [0.9, 0.02])


if __name__ == "__main__":
    unittest.main()
