"""The standard normal CDF shared by the crypto probability paths.

crypto_5m._normal_cdf is

    0.5 * (1 + erf(z / sqrt(2)))

and it is the only one. crypto_5m uses it to turn a z-score into
probability_up in two places, crypto_kalshi calls it for the d2 term, and
scripts/crypto_kalshi_edge reports it. probability_up then feeds
_strategy_from_edge, which decides whether to trade and how much.

The sqrt(2) was unheld: removing it passed all 1,958 tests while shifting
every probability upward.

    z     correct   without sqrt(2)
    0.5   0.6915    0.7602   (+6.9pp)
    1.0   0.8413    0.9214   (+8.0pp)
    1.5   0.9332    0.9831   (+5.0pp)

This is the second implementation of the same curve with the same gap --
weather_research had it too (#573) -- so the tests are written against
the textbook values rather than the formula, and both places now hold.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.crypto_5m import _normal_cdf  # noqa: E402


class TheTextbookValuesTests(unittest.TestCase):
    """68-95-99.7. These cannot drift with the implementation."""

    def test_zero_is_the_median(self):
        self.assertAlmostEqual(_normal_cdf(0.0), 0.5, places=10)

    def test_one_sigma(self):
        """0.8413, not 0.9214. The difference is the sqrt(2)."""
        self.assertAlmostEqual(_normal_cdf(1.0), 0.8413, places=4)

    def test_two_sigma(self):
        self.assertAlmostEqual(_normal_cdf(2.0), 0.9772, places=4)

    def test_three_sigma(self):
        self.assertAlmostEqual(_normal_cdf(3.0), 0.9987, places=4)

    def test_the_central_intervals_are_the_familiar_ones(self):
        self.assertAlmostEqual(_normal_cdf(1.0) - _normal_cdf(-1.0), 0.6827, places=4)
        self.assertAlmostEqual(_normal_cdf(2.0) - _normal_cdf(-2.0), 0.9545, places=4)

    def test_the_95_percent_quantile_is_at_1_96(self):
        self.assertAlmostEqual(_normal_cdf(1.959963984540054), 0.975, places=5)


class ShapeTests(unittest.TestCase):
    def test_it_is_symmetric(self):
        for z in (0.25, 1.0, 2.5, 4.0):
            with self.subTest(z=z):
                self.assertAlmostEqual(_normal_cdf(z) + _normal_cdf(-z), 1.0, places=10)

    def test_it_is_monotonic(self):
        values = [_normal_cdf(z / 10.0) for z in range(-40, 41)]
        self.assertEqual(values, sorted(values))

    def test_it_stays_within_zero_and_one(self):
        for z in (-40.0, -6.0, 0.0, 6.0, 40.0):
            with self.subTest(z=z):
                self.assertGreaterEqual(_normal_cdf(z), 0.0)
                self.assertLessEqual(_normal_cdf(z), 1.0)

    def test_the_tails_saturate_rather_than_overflow(self):
        self.assertAlmostEqual(_normal_cdf(-40.0), 0.0, places=10)
        self.assertAlmostEqual(_normal_cdf(40.0), 1.0, places=10)


class ItIsTheSameCurveAsTheWeatherOneTests(unittest.TestCase):
    """Two implementations, one distribution. They must not diverge."""

    def test_a_standardised_weather_bracket_matches(self):
        from analyzing_llm_rationale.weather_research import (
            calculate_bracket_probability,
        )

        mean, std = 70.0, 2.0
        for z in (0.5, 1.0, 2.0):
            with self.subTest(z=z):
                weather = calculate_bracket_probability(
                    mean, {"strike_type": "less_than", "strike_f": mean + z * std},
                    uncertainty_std_f=std,
                )["model_probability"]
                self.assertAlmostEqual(weather, round(_normal_cdf(z), 3), places=3)


if __name__ == "__main__":
    unittest.main()
