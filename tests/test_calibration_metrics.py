"""Expected calibration error, including the confident forecast.

ece bins by `int(confidence * bins)` and clamps the result to the last
bin. The clamp is not cosmetic: a confidence of exactly 1.0 gives
int(1.0 * 10) = 10, one past the end of a ten-element list, so without it
the function raises IndexError on a model that says 100%.

Nothing tested that. A forecaster stating certainty is ordinary input,
and this is the metric that grades how honest such statements are.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.metrics import (  # noqa: E402
    Example,
    accuracy,
    brier_score,
    ece,
)


def _ex(answer: str, confidence: float, target: int) -> Example:
    return Example(predicted_answer=answer, confidence=confidence, target=target)


class TheTopBinTests(unittest.TestCase):
    def test_total_certainty_does_not_fall_off_the_end(self):
        """int(1.0 * bins) is bins, and the last valid index is bins - 1."""
        result = ece([_ex("yes", 1.0, 1), _ex("no", 0.5, 0)])
        self.assertTrue(math.isfinite(result))

    def test_certainty_that_was_right_and_wrong_both_bin(self):
        for target in (0, 1):
            with self.subTest(target=target):
                self.assertTrue(math.isfinite(ece([_ex("yes", 1.0, target)])))

    def test_the_bin_index_is_clamped_not_wrapped(self):
        """A 100% forecast belongs in the top bin, not the bottom one.

        Wrapping (`% bins`) sends it to bin 0, beside the least confident
        forecasts, where averaging hides it. A single example cannot show
        this -- the bin only groups, so with one entry the average is the
        same wherever it lands. Two entries at opposite ends can:

            confidence 1.00 and wrong, confidence 0.02 and right
            clamped  ECE = 0.990   wrapped  ECE = 0.010

        Wrapping reports near-perfect calibration for a model that was
        certain and mistaken, which is the single worst thing this metric
        is meant to catch.
        """
        certain_and_wrong = _ex("yes", 1.0, 0)
        unsure_and_right = _ex("yes", 0.02, 1)

        self.assertAlmostEqual(ece([certain_and_wrong, unsure_and_right]), 0.99, places=6)

    def test_certainty_alone_scores_as_you_would_expect(self):
        self.assertAlmostEqual(ece([_ex("yes", 1.0, 1)]), 0.0)
        self.assertAlmostEqual(ece([_ex("yes", 1.0, 0)]), 1.0)


class CalibrationTests(unittest.TestCase):
    def test_a_perfectly_calibrated_set_scores_zero(self):
        """Half of the 50% calls right is exactly what 50% claims."""
        examples = [_ex("yes", 0.5, 1), _ex("yes", 0.5, 0)]
        self.assertAlmostEqual(ece(examples), 0.0)

    def test_overconfidence_scores_above_zero(self):
        examples = [_ex("yes", 0.9, 1), _ex("yes", 0.9, 0)]
        self.assertGreater(ece(examples), 0.0)

    def test_an_empty_set_has_no_calibration_error(self):
        self.assertTrue(math.isnan(ece([])))

    def test_the_error_is_weighted_by_bin_population(self):
        """A wrong bin holding one forecast must not outweigh a right one
        holding many."""
        many_right = [_ex("yes", 0.5, 1), _ex("yes", 0.5, 0)] * 50
        one_wrong = [_ex("yes", 0.9, 0)]
        blended = ece(many_right + one_wrong)
        self.assertLess(blended, ece(one_wrong))


class ScoreShapeTests(unittest.TestCase):
    def test_brier_is_squared_error(self):
        """0.5 confidence on a yes that resolved yes: (0.5 - 1)^2 = 0.25."""
        self.assertAlmostEqual(brier_score([_ex("yes", 0.5, 1)]), 0.25)

    def test_brier_rewards_confident_correctness(self):
        self.assertLess(
            brier_score([_ex("yes", 0.9, 1)]), brier_score([_ex("yes", 0.6, 1)]),
        )

    def test_accuracy_is_a_rate_not_a_count(self):
        examples = [_ex("yes", 0.8, 1), _ex("yes", 0.8, 0)]
        self.assertAlmostEqual(accuracy(examples), 0.5)


if __name__ == "__main__":
    unittest.main()
