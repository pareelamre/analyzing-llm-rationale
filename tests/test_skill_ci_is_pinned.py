"""The interval behind "this disagreement has proven edge".

_skill_ci puts a 95% CI on skill-vs-market and sets `skill_significant`
only when the lower bound clears zero. Its own docstring calls that "the
gate that turns 'we disagree' into 'this disagreement has proven edge'",
and edge_calibration reports it per edge bucket, so the buckets can be
small.

The sample variance used Bessel's correction and nothing held it there.
Dividing by n instead of n-1 narrows the interval, which biases the gate
toward claiming significance -- the wrong direction for a claim about
having beaten the market.

At three snapshots with diffs [0.0343, 0.004, 0.0116]:

    n-1 (correct)  low = -0.00121  ->  not significant
    n   (mutated)  low = +0.00207  ->  significant

so the mutation publishes proven edge where the honest computation says
the result is not distinguishable from noise.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.track_record_live import _skill_ci  # noqa: E402


def _rows(diffs):
    """Snapshots whose market_brier - model_brier equals each diff."""
    return [{"model_brier": 0.2000, "market_brier": round(0.2000 + d, 6)} for d in diffs]


class BesselsCorrectionTests(unittest.TestCase):
    def test_a_small_sample_is_not_called_significant(self):
        """The case that separates n-1 from n. Without the correction this
        reports proven edge on three snapshots."""
        result = _skill_ci(_rows([0.0343, 0.004, 0.0116]))
        self.assertFalse(result["skill_significant"])
        self.assertLess(result["skill_ci_low"], 0.0)

    def test_the_interval_matches_the_sample_variance(self):
        diffs = [0.0343, 0.004, 0.0116]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        result = _skill_ci(_rows(diffs))
        self.assertAlmostEqual(result["skill_ci_low"], round(mean - 1.96 * se, 4), places=4)
        self.assertAlmostEqual(result["skill_ci_high"], round(mean + 1.96 * se, 4), places=4)

    def test_the_correction_matters_least_when_the_sample_is_large(self):
        """Why this went unnoticed: at the published n it is invisible."""
        diffs = [0.01] * 2000 + [0.02] * 835
        n = len(diffs)
        mean = sum(diffs) / n
        with_bessel = (sum((d - mean) ** 2 for d in diffs) / (n - 1) / n) ** 0.5
        without = (sum((d - mean) ** 2 for d in diffs) / n / n) ** 0.5
        self.assertLess(abs(with_bessel - without) / with_bessel, 0.001)


class SignificanceGateTests(unittest.TestCase):
    def test_significance_requires_the_lower_bound_above_zero(self):
        beat = _skill_ci(_rows([0.05] * 30))
        self.assertGreater(beat["skill_ci_low"], 0.0)
        self.assertTrue(beat["skill_significant"])

    def test_losing_to_the_market_is_never_significant(self):
        """One-sided by construction: a confidently worse model is not
        'significant', it is bad."""
        lost = _skill_ci(_rows([-0.05] * 30))
        self.assertLess(lost["skill_ci_high"], 0.0)
        self.assertFalse(lost["skill_significant"])

    def test_straddling_zero_is_not_significant(self):
        straddle = _skill_ci(_rows([0.05, -0.05] * 15))
        self.assertLess(straddle["skill_ci_low"], 0.0)
        self.assertFalse(straddle["skill_significant"])

    def test_the_interval_uses_the_95_percent_multiplier(self):
        diffs = [0.05] * 30
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        result = _skill_ci(_rows(diffs))
        self.assertAlmostEqual(result["skill_ci_high"] - result["skill_ci_low"],
                               round(2 * 1.96 * se, 4), places=3)


class TooFewRowsTests(unittest.TestCase):
    def test_no_rows_gives_no_interval(self):
        result = _skill_ci([])
        self.assertIsNone(result["skill_ci_low"])
        self.assertFalse(result["skill_significant"])

    def test_a_single_row_gives_no_interval(self):
        """One observation has no sample variance; n-1 would divide by zero."""
        result = _skill_ci(_rows([0.05]))
        self.assertIsNone(result["skill_ci_low"])
        self.assertIsNone(result["skill_ci_high"])
        self.assertFalse(result["skill_significant"])


if __name__ == "__main__":
    unittest.main()
