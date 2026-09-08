"""The normal CDF behind every published weather bracket probability.

calculate_bracket_probability integrates a Gaussian over a contract strike:

    0.5 * (1 + erf((x - mean) / (std * sqrt(2))))

The sqrt(2) is what makes that the normal CDF rather than a narrower
curve, and nothing held it there. Dropping it does not raise or return
anything obviously wrong -- it returns confident numbers that are simply
too high:

    strike     correct   without sqrt(2)
    +0.5 sigma  0.6915    0.7602   (+6.9pp)
    +1.0 sigma  0.8413    0.9214   (+8.0pp)
    +1.5 sigma  0.9332    0.9831   (+5.0pp)

These probabilities are published on /market/weather-radar and fed to
sizing, so an eight-point overstatement at one sigma is a real edge that
is not there.

The textbook values anchor the tests: a normal CDF gives 0.8413 at one
sigma and 0.9772 at two. Those cannot drift with the implementation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.weather_research import (  # noqa: E402
    calculate_bracket_probability,
)

MEAN = 70.0
STD = 2.0


def _less_than(strike: float, std: float = STD):
    return calculate_bracket_probability(
        MEAN, {"strike_type": "less_than", "strike_f": strike}, uncertainty_std_f=std,
    )["model_probability"]


class TheCurveIsTheNormalCdfTests(unittest.TestCase):
    def test_the_mean_splits_the_distribution(self):
        self.assertAlmostEqual(_less_than(MEAN), 0.5, places=3)

    def test_one_sigma_is_the_textbook_value(self):
        """0.8413, not 0.9214. The difference is the sqrt(2)."""
        self.assertAlmostEqual(_less_than(MEAN + STD), 0.841, places=3)

    def test_two_sigma_is_the_textbook_value(self):
        self.assertAlmostEqual(_less_than(MEAN + 2 * STD), 0.977, places=3)

    def test_it_is_symmetric_about_the_mean(self):
        below = _less_than(MEAN - STD)
        above = _less_than(MEAN + STD)
        self.assertAlmostEqual(below + above, 1.0, places=3)

    def test_the_width_follows_the_stated_uncertainty(self):
        """A wider forecast is less sure about the same strike."""
        tight = _less_than(MEAN + 2.0, std=1.0)
        loose = _less_than(MEAN + 2.0, std=4.0)
        self.assertGreater(tight, loose)


class StrikeTypesTests(unittest.TestCase):
    def _prob(self, spec, std: float = STD):
        return calculate_bracket_probability(MEAN, spec, uncertainty_std_f=std)["model_probability"]

    def test_greater_than_complements_less_than(self):
        strike = MEAN + STD
        below = self._prob({"strike_type": "less_than", "strike_f": strike})
        above = self._prob({"strike_type": "greater_than", "strike_f": strike})
        self.assertAlmostEqual(below + above, 1.0, places=3)

    def test_a_between_bracket_is_the_difference_of_the_bounds(self):
        one_sigma_either_side = self._prob({
            "strike_type": "between",
            "strike_low_f": MEAN - STD, "strike_high_f": MEAN + STD,
        })
        self.assertAlmostEqual(one_sigma_either_side, 0.683, places=2)

    def test_an_inverted_bracket_is_not_negative(self):
        inverted = self._prob({
            "strike_type": "between",
            "strike_low_f": MEAN + STD, "strike_high_f": MEAN - STD,
        })
        self.assertGreaterEqual(inverted, 0.0)

    def test_an_unknown_strike_type_says_it_does_not_know(self):
        self.assertAlmostEqual(self._prob({"strike_type": "sideways"}), 0.5, places=3)


class UncertaintyFloorTests(unittest.TestCase):
    def test_zero_uncertainty_does_not_divide_by_zero(self):
        """A caller reporting no uncertainty is wrong, not fatal."""
        result = calculate_bracket_probability(
            MEAN, {"strike_type": "less_than", "strike_f": MEAN + 1.0},
            uncertainty_std_f=0.0,
        )
        self.assertGreater(result["uncertainty_std_f"], 0.0)
        self.assertTrue(0.0 < result["model_probability"] < 1.0)

    def test_a_negative_uncertainty_is_floored_too(self):
        result = calculate_bracket_probability(
            MEAN, {"strike_type": "less_than", "strike_f": MEAN},
            uncertainty_std_f=-3.0,
        )
        self.assertGreater(result["uncertainty_std_f"], 0.0)


class ReportedRangeTests(unittest.TestCase):
    def test_certainty_is_never_published_as_one(self):
        far = calculate_bracket_probability(
            MEAN, {"strike_type": "less_than", "strike_f": MEAN + 100.0},
        )["model_probability"]
        self.assertLessEqual(far, 0.999)
        self.assertGreater(far, 0.99)

    def test_impossibility_is_never_published_as_zero(self):
        far = calculate_bracket_probability(
            MEAN, {"strike_type": "less_than", "strike_f": MEAN - 100.0},
        )["model_probability"]
        self.assertGreaterEqual(far, 0.001)

    def test_the_interval_is_the_95_percent_one(self):
        result = calculate_bracket_probability(
            MEAN, {"strike_type": "less_than", "strike_f": MEAN}, uncertainty_std_f=STD,
        )
        low, high = result["confidence_interval_95_f"]
        self.assertAlmostEqual(high - low, 2 * 1.96 * STD, places=1)


if __name__ == "__main__":
    unittest.main()
