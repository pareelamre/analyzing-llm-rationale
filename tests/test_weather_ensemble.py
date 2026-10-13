"""Tests for ensemble-based weather strike probabilities.

Covers the day-window fix (local-day grouping, not UTC [:24] slicing), the
ensemble member histogram, station-bias application, and the cross-source
divergence no-trade signal.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.weather_research import (
    _ensemble_strike_probability,
    _fetch_open_meteo_forecast,
    parse_market_strike,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class EnsembleStrikeProbabilityTests(unittest.TestCase):
    def _ensemble(self, maxima):
        n = len(maxima)
        return {
            "model": "gfs025",
            "member_count": n,
            "ensemble_mean_high_f": round(sum(maxima) / n, 1),
            "ensemble_std_f": 1.0,
            "member_maxima_f": list(maxima),
        }

    def test_less_than_strike_counts_members(self):
        ens = self._ensemble([68.0, 69.0, 70.5, 72.0, 73.0, 74.0,
                              68.5, 69.5, 70.0, 71.5, 72.5, 75.0])
        prob = _ensemble_strike_probability(
            ens, {"strike_type": "less_than", "strike_f": 71.0})
        # 6 of 12 members strictly below 71 (70.5, 70.0, 68.0, 69.0, 68.5, 69.5)
        self.assertEqual(prob["model_probability"], 0.5)
        self.assertEqual(prob["method"], "ensemble_member_histogram")
        self.assertEqual(prob["member_count"], 12)

    def test_bias_offset_shifts_threshold_crossing(self):
        ens = self._ensemble([70.4, 70.6, 70.8, 71.2, 71.4, 71.6,
                              70.2, 70.9, 71.1, 71.3, 71.5, 71.8])
        no_bias = _ensemble_strike_probability(
            ens, {"strike_type": "less_than", "strike_f": 71.0}, bias_offset_f=0.0)
        with_bias = _ensemble_strike_probability(
            ens, {"strike_type": "less_than", "strike_f": 71.0}, bias_offset_f=-0.5)
        # A negative offset (cooler station) must raise P(high < 71)
        self.assertGreater(with_bias["model_probability"],
                           no_bias["model_probability"])
        self.assertEqual(with_bias["bias_offset_applied_f"], -0.5)

    def test_greater_than_strike(self):
        ens = self._ensemble([69.0, 69.5, 70.0, 70.5, 71.5, 72.0,
                              72.5, 73.0, 73.5, 74.0, 74.5, 75.0])
        prob = _ensemble_strike_probability(
            ens, {"strike_type": "greater_than_or_equal", "strike_f": 71.0})
        # 8 of 12 members at or above 71
        self.assertEqual(prob["model_probability"], 0.667)

    def test_between_strike(self):
        ens = self._ensemble([69.0, 70.0, 71.0, 72.0, 73.0, 74.0,
                              75.0, 76.0, 77.0, 78.0, 79.0, 80.0])
        prob = _ensemble_strike_probability(
            ens, {"strike_type": "between", "strike_low_f": 71.0, "strike_high_f": 75.0})
        # 5 of 12 members in [71, 75]
        self.assertEqual(prob["model_probability"], 0.417)

    def test_too_few_members_returns_none(self):
        ens = self._ensemble([70.0, 71.0, 72.0])
        self.assertIsNone(_ensemble_strike_probability(
            ens, {"strike_type": "less_than", "strike_f": 71.0}))

    def test_unknown_strike_type_returns_none(self):
        ens = self._ensemble([70.0] * 12)
        self.assertIsNone(_ensemble_strike_probability(
            ens, {"strike_type": "exactly", "strike_f": 71.0}))


class DayWindowTests(unittest.TestCase):
    def test_local_day_grouping_not_utc_slice(self):
        """The deterministic fetch must group by local calendar day.

        Regression: the old implementation sliced times[:24] on a UTC array,
        which spans two local days and can miss the afternoon peak.
        """
        # 48 hours starting 2026-10-13T00:00 UTC. In America/New_York with
        # timezone=auto, Open-Meteo returns local-midnight-aligned hours.
        hours = [f"2026-10-13T{h:02d}:00" for h in range(24)] + \
                [f"2026-10-14T{h:02d}:00" for h in range(24)]
        # Day 1 peaks at 69F at 3 PM local; day 2 peaks at 80F.
        temps = [55.0 + (10.0 if 13 <= h % 24 <= 19 else 0.0) for h in range(24)]
        temps[15] = 69.0  # 3 PM day 1
        temps += [60.0 + (18.0 if 13 <= h % 24 <= 19 else 0.0) for h in range(24)]
        temps[24 + 15] = 80.0  # 3 PM day 2

        calls = []

        def fake_get(url, timeout=None):
            calls.append(url)
            return _FakeResponse({"hourly": {"time": hours, "temperature_2m": temps}})

        result = _fetch_open_meteo_forecast(fake_get, 40.7829, -73.9654)
        self.assertIsNotNone(result)
        # The timezone=auto parameter must be requested.
        self.assertIn("timezone=auto", calls[0])
        # The first FULL local day is the target: high 69.0, not 80.0.
        self.assertEqual(result["forecast_day_local"], "2026-10-13")
        self.assertEqual(result["projected_high_f"], 69.0)
        self.assertEqual(result["projected_low_f"], 55.0)

    def test_trailing_partial_day_dropped(self):
        hours = [f"2026-10-13T{h:02d}:00" for h in range(24)] + \
                ["2026-10-14T00:00", "2026-10-14T01:00"]
        temps = [60.0] * 24 + [61.0, 62.0]

        def fake_get(url, timeout=None):
            return _FakeResponse({"hourly": {"time": hours, "temperature_2m": temps}})

        result = _fetch_open_meteo_forecast(fake_get, 40.7829, -73.9654)
        self.assertEqual(result["forecast_day_local"], "2026-10-13")
        self.assertEqual(result["projected_high_f"], 60.0)


class StrikeParsingTests(unittest.TestCase):
    def test_less_than_71(self):
        spec = parse_market_strike({
            "title": "Will the maximum temperature be <71° on Oct 13, 2026?",
            "subtitle": "70° or below",
        })
        self.assertIsNotNone(spec)
        self.assertEqual(spec["strike_type"], "less_than")
        self.assertEqual(spec["strike_f"], 71.0)


if __name__ == "__main__":
    unittest.main()
