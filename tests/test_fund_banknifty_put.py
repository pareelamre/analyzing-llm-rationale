from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("fund_banknifty_put_eval", _ROOT / "scripts" / "fund_banknifty_put_eval.py")
bank = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bank)


def _series(n: int, scale: float, start=date(2008, 1, 1)) -> list[tuple[date, float]]:
    days = [start + timedelta(days=i) for i in range(n)]
    return [(d, 10_000 * math.exp(0.0002 * i + scale * 0.01 * math.sin(i * 1.7))) for i, d in enumerate(days)]


class RatioTests(unittest.TestCase):
    def test_the_vol_ratio_is_bank_niftys_realized_volatility_over_niftys(self):
        nif, bnk = _series(200, 1.0), _series(200, 1.5)
        self.assertAlmostEqual(bank.vol_ratio(bnk, nif, nif[-1][0]), 1.5, delta=0.05)
        self.assertIsNone(bank.vol_ratio(bnk[:30], nif, nif[-1][0]))  # under 60 sessions

    def test_the_ratio_uses_only_the_past(self):
        nif, bnk = _series(300, 1.0), _series(300, 1.5)
        day = nif[150][0]
        wild = bnk[:151] + [(d, c * (3 if i % 2 else 1)) for i, (d, c) in enumerate(bnk[151:])]
        self.assertEqual(bank.vol_ratio(bnk, nif, day), bank.vol_ratio(wild, nif, day))


class BothSoldTests(unittest.TestCase):
    def test_correlation_and_the_combined_path(self):
        a = [{"day": f"2010-01-{i + 1:02d}", "return": r} for i, r in enumerate([0.2, -1.0, 0.2, 0.1])]
        b = [{"day": f"2010-01-{i + 1:02d}", "return": r} for i, r in enumerate([0.2, -1.0, 0.2, 0.1])]
        out = bank.both_sold(a, b)
        self.assertAlmostEqual(out["rho"], 1.0)
        self.assertEqual((out["decisions"], out["both_lost"]), (4, 1))
        self.assertAlmostEqual(out["worst_decision_at_5pct_each"], -0.10)


class ReportTests(unittest.TestCase):
    def test_the_committed_report_follows_the_pre_registered_rule(self):
        report = json.loads((_ROOT / "reports" / "fund_banknifty_put_eval.json").read_text())
        put = report["put_spread"]
        self.assertIn("pre-registered 2026-11-16", report["rule"])
        self.assertGreaterEqual(put["trades"], 60)
        self.assertEqual(put["verdict"], "GO" if put["mean_return_ci"][0] > 0 else "NO-GO")
        self.assertTrue(0 <= report["description_only"]["both_sold"]["rho"] <= 1)


if __name__ == "__main__":
    unittest.main()
