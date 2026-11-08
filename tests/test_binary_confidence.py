from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server  # noqa: E402


class BinaryConfidenceTests(unittest.TestCase):
    def _yes_probability(self, answer, confidence):
        confidence = server._confidence_in_answer(answer, confidence)
        return server._model_probability_for_market("binary", answer, confidence, [], "Yes")

    def test_a_low_confidence_belongs_to_the_other_side(self):
        # The on-device model wrote "No" with 0.02 for markets its rationale
        # called near impossible; that must read as 2% Yes, not 98%.
        self.assertAlmostEqual(self._yes_probability("No", 0.02), 0.02)
        self.assertAlmostEqual(self._yes_probability("No", 0.28), 0.28)
        self.assertAlmostEqual(self._yes_probability("Yes", 0.3), 0.7)

    def test_a_coherent_confidence_is_unchanged(self):
        self.assertAlmostEqual(self._yes_probability("No", 0.96), 0.04)
        self.assertAlmostEqual(self._yes_probability("Yes", 0.8), 0.8)
        self.assertEqual(server._confidence_in_answer("Yes", 0.5), 0.5)

    def test_non_binary_answers_pass_through(self):
        self.assertEqual(server._confidence_in_answer("Option A", 0.2), 0.2)
        self.assertIsNone(server._confidence_in_answer("No", None))


if __name__ == "__main__":
    unittest.main()
