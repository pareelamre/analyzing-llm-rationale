from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fund_correlation_scan", _ROOT / "scripts" / "fund_correlation_scan.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

D = date(2026, 11, 3)  # a Tuesday


class ScanTests(unittest.TestCase):
    def test_a_predictor_is_only_ever_the_value_dated_before_the_session(self):
        values = {D - timedelta(days=1): 1.0, D: 2.0, D + timedelta(days=1): 3.0}
        self.assertEqual(cs.known_before(values, [D, D + timedelta(days=1)]), {D: 1.0, D + timedelta(days=1): 2.0})
        self.assertEqual(cs.known_before({D - timedelta(days=9): 1.0}, [D]), {})  # too stale to count

    def test_changes_and_correlations(self):
        series = [(D, 100.0), (D + timedelta(days=1), 110.0)]
        self.assertAlmostEqual(cs.log_change(series)[D + timedelta(days=1)], math.log(1.1))
        self.assertEqual(cs.diff(series)[D + timedelta(days=1)], 10.0)
        xs = [float(i) for i in range(100)]
        r, t = cs.corr_t(xs, [2 * x + 1 for x in xs])
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual(cs.corr_t(xs[:10], xs[:10]), (None, None))

    def test_a_test_needs_discovery_and_holdout_and_then_a_profit_after_costs(self):
        days = [cs.START + timedelta(days=i) for i in range(0, 6000, 1) if (cs.START + timedelta(days=i)).weekday() < 5]
        x = {d: math.sin(i * 1.7) for i, d in enumerate(days)}
        target = {d: 0.004 * x[d] for d in days}  # the target follows the predictor exactly
        [row] = cs.scan([("p", x, "t", "index", target)])
        self.assertTrue(row["candidate"] and row["confirmed"])
        self.assertEqual(row["rule"]["verdict"], "GO")
        noise = {d: 0.004 * math.sin(i * 0.37 + 1) for i, d in enumerate(days)}
        [none] = cs.scan([("p", x, "t", "index", noise)])
        self.assertFalse(none["confirmed"])
        self.assertNotIn("rule", none)

    def test_the_spread_follows_the_trackers_hedge_and_z_windows(self):
        days = [D + timedelta(days=i) for i in range(400)]
        a = [(d, 100 * math.exp(0.001 * i + 0.01 * math.sin(i))) for i, d in enumerate(days)]
        b = [(d, 50 * math.exp(0.001 * i)) for i, d in enumerate(days)]
        z, spread = cs.spread_z(a, b)
        self.assertEqual(min(z), days[cs.HEDGE_WINDOW])
        self.assertTrue(all(abs(v) < 5 for v in z.values()))


if __name__ == "__main__":
    unittest.main()
