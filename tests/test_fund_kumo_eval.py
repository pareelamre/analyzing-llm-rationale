from __future__ import annotations

import importlib.util
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_kumo_eval", ROOT / "scripts" / "fund_kumo_eval.py")
ke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ke)


def _days(n):
    return [date(2000, 1, 3) + timedelta(days=k) for k in range(n)]


class ScheduleTests(unittest.TestCase):
    def test_decisions_start_with_a_full_context_and_do_not_overlap(self):
        got = ke.decisions(600, 21)
        self.assertEqual(got[0], ke.SEQ_LEN - 1)
        self.assertTrue(all(b - a == 21 for a, b in zip(got, got[1:])))
        self.assertLess(got[-1] + 21, 600)


class BaselineTests(unittest.TestCase):
    def test_mean_reversion_uses_only_changes_complete_by_the_day(self):
        series = [((k * 37) % 11) / 10.0 for k in range(700)]
        i = 600
        before = ke.mean_reversion(series, i)
        changed = series[: i + 1] + [99.0] * (len(series) - i - 1)
        self.assertAlmostEqual(ke.mean_reversion(changed, i), before)

    def test_mean_reversion_finds_a_reverting_series(self):
        # A series that always closes half its gap to the mean over the horizon.
        series, x = [], 0.0
        for k in range(700):
            x = 0.97 * x + (0.3 if k % 7 == 0 else -0.05)
            series.append(x)
        pred = ke.mean_reversion(series, 650)
        gap = series[650] - sum(series[591:651]) / 60
        self.assertLess(pred * gap, 0)  # forecasts a move back toward the mean


class ScoreTests(unittest.TestCase):
    def test_a_perfect_forecaster_beats_no_change(self):
        n = 700
        values = [0.001 * k for k in range(n)]

        def perfect(days, window, horizon):
            last = window[-1]
            return [last + 0.001 * (h + 1) for h in range(horizon)]
        rows = ke.run_series("x", _days(n), values, 21, perfect)
        s = ke.score(rows, "kumo", "no_change")
        self.assertLess(s["mean_abs_diff"], 0)
        self.assertLess(s["mean_abs_diff_ci"][1], 0)
        self.assertEqual(ke.hit_rate(rows, "kumo"), 1.0)

    def test_the_forecaster_sees_only_the_context_window(self):
        n = 600
        seen = []

        def spy(days, window, horizon):
            seen.append((days[-1], len(window)))
            return [window[-1]] * horizon
        rows = ke.run_series("x", _days(n), [float(k) for k in range(n)], 21, spy)
        self.assertEqual(seen[0], (_days(n)[ke.SEQ_LEN - 1], ke.SEQ_LEN))
        self.assertEqual(rows[0]["actual"], 21.0)

    def test_a_failing_darr_call_is_left_out(self):
        def flat(days, window, horizon):
            return [window[-1]] * horizon

        def broken(days, window, horizon):
            raise ValueError("too little context")
        rows = ke.run_series("x", _days(560), [float(k) for k in range(560)], 20, flat, darr=broken, pair=True)
        self.assertIsNone(rows[0]["kumo_darr"])
        self.assertEqual(rows[0]["darr_error"], "ValueError")
        self.assertEqual(ke.score(rows, "kumo_darr", "no_change")["decisions"], 0)


if __name__ == "__main__":
    unittest.main()
