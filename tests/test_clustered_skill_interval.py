"""The clustered confidence interval on Brier skill.

market_clustered_brier_skill_interval treats markets, not snapshots, as
the independent unit -- it averages the per-snapshot Brier differences
within each market first, then puts a normal interval around the mean of
those market means. That clustering is the point: repeated snapshots of
one market are not independent evidence, and counting them as if they
were narrows the interval on nothing.

Three parts of it were unpinned, each failing in a different direction:

    n_markets - 1        -> n_markets    interval too narrow
    sqrt(variance / n)   -> sqrt(var)    interval too wide, by sqrt(n)
    n_markets >= 2       -> >= 1         divides by zero on one market

The result is published in static/forecast_evaluation.json.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.forecast_evaluation import (  # noqa: E402
    ResolvedForecast,
    market_clustered_brier_skill_interval,
)

WHEN = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _row(market_id: str, model_p: float, market_p: float, outcome: int, idx: int = 0):
    return ResolvedForecast(
        forecast_id=f"{market_id}-{idx}", platform="kalshi", market_id=market_id,
        model="m", forecasted_at=WHEN, resolved_at=WHEN,
        model_probability=model_p, market_probability=market_p, outcome=outcome,
    )


class TheIntervalArithmeticTests(unittest.TestCase):
    def _skills_and_result(self, rows):
        return market_clustered_brier_skill_interval(rows)

    def test_the_interval_is_the_mean_plus_or_minus_z_standard_errors(self):
        """Computed independently from the per-market skills."""
        rows = [
            _row("a", 0.9, 0.5, 1), _row("b", 0.8, 0.5, 1),
            _row("c", 0.7, 0.5, 1), _row("d", 0.4, 0.5, 1),
        ]
        result = self._skills_and_result(rows)

        # Each market contributes market_error - model_error, with the market
        # at 0.5 and the outcome 1 throughout.
        skills = [
            (0.5 - 1) ** 2 - (model_p - 1) ** 2
            for model_p in (0.9, 0.8, 0.7, 0.4)
        ]
        n = len(skills)
        mean = sum(skills) / n
        variance = sum((s - mean) ** 2 for s in skills) / (n - 1)
        se = math.sqrt(variance / n)

        self.assertAlmostEqual(result["mean_skill"], mean, places=10)
        self.assertAlmostEqual(result["standard_error"], se, places=10)
        self.assertAlmostEqual(result["lower"], mean - 1.96 * se, places=10)
        self.assertAlmostEqual(result["upper"], mean + 1.96 * se, places=10)

    def test_the_standard_error_is_of_the_mean_not_the_sample(self):
        """sqrt(var/n), not sqrt(var). The difference is a factor of sqrt(n)."""
        rows = [_row(f"m{i}", 0.9 if i % 2 else 0.4, 0.5, 1) for i in range(16)]
        result = market_clustered_brier_skill_interval(rows)

        skills = [(0.5 - 1) ** 2 - ((0.9 if i % 2 else 0.4) - 1) ** 2 for i in range(16)]
        n = len(skills)
        mean = sum(skills) / n
        variance = sum((s - mean) ** 2 for s in skills) / (n - 1)

        self.assertAlmostEqual(result["standard_error"], math.sqrt(variance / n), places=10)
        self.assertLess(result["standard_error"], math.sqrt(variance) / 3)

    def test_more_markets_narrow_the_interval(self):
        """The property the standard error exists to express."""
        def width(count):
            rows = [_row(f"m{i}", 0.9 if i % 2 else 0.4, 0.5, 1) for i in range(count)]
            out = market_clustered_brier_skill_interval(rows)
            return out["upper"] - out["lower"]

        self.assertLess(width(40), width(10))


class TheClusteringTests(unittest.TestCase):
    def test_repeated_snapshots_of_one_market_count_once(self):
        """Twenty snapshots of a single market are one independent unit."""
        rows = [_row("a", 0.9, 0.5, 1, idx=i) for i in range(20)]
        result = market_clustered_brier_skill_interval(rows)
        self.assertEqual(result["n_forecasts"], 20)
        self.assertEqual(result["n_markets"], 1)

    def test_a_single_market_gets_no_interval(self):
        """One unit has no sample variance; n-1 would divide by zero."""
        rows = [_row("a", 0.9, 0.5, 1, idx=i) for i in range(20)]
        result = market_clustered_brier_skill_interval(rows)
        self.assertIsNone(result["standard_error"])
        self.assertIsNone(result["lower"])
        self.assertIsNone(result["upper"])
        self.assertIsNotNone(result["mean_skill"])

    def test_snapshots_within_a_market_are_averaged_before_clustering(self):
        """One market with two snapshots equals one market with their mean."""
        pair = market_clustered_brier_skill_interval(
            [_row("a", 0.9, 0.5, 1, idx=0), _row("a", 0.7, 0.5, 1, idx=1),
             _row("b", 0.6, 0.5, 1)]
        )
        skill_a = (((0.5 - 1) ** 2 - (0.9 - 1) ** 2) + ((0.5 - 1) ** 2 - (0.7 - 1) ** 2)) / 2
        skill_b = (0.5 - 1) ** 2 - (0.6 - 1) ** 2
        self.assertAlmostEqual(pair["mean_skill"], (skill_a + skill_b) / 2, places=10)

    def test_no_forecasts_reports_nothing_rather_than_zero(self):
        result = market_clustered_brier_skill_interval([])
        self.assertEqual(result["n_markets"], 0)
        self.assertIsNone(result["mean_skill"])
        self.assertIsNone(result["lower"])


class TheReportedMethodTests(unittest.TestCase):
    def test_the_confidence_level_is_only_claimed_for_the_matching_z(self):
        rows = [_row(f"m{i}", 0.9 if i % 2 else 0.4, 0.5, 1) for i in range(6)]
        self.assertEqual(
            market_clustered_brier_skill_interval(rows)["confidence_level"], 0.95,
        )
        self.assertIsNone(
            market_clustered_brier_skill_interval(rows, z_score=1.64)["confidence_level"],
        )

    def test_a_non_positive_z_is_refused(self):
        with self.assertRaises(ValueError):
            market_clustered_brier_skill_interval([], z_score=0.0)


if __name__ == "__main__":
    unittest.main()
