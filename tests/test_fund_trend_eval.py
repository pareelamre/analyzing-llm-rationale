from __future__ import annotations

import importlib.util
import math
import random
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_trend_eval", ROOT / "scripts" / "fund_trend_eval.py")
te = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = te
_spec.loader.exec_module(te)


def _sessions(start: date, end: date, holidays=()) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            out.append(d)
        d += timedelta(days=1)
    return out


class DataTests(unittest.TestCase):
    def test_eia_weeks_become_dated_prices_and_blanks_are_skipped(self):
        html = ("<tr>\n <td class='B6'>&nbsp;&nbsp;1983 Apr-18 to Apr-22</td>\n <td class='B3'>30.75</td>\n"
                " <td class='B3'>30.75</td>\n <td class='B3'>30.70</td>\n <td class='B3'>1,030.68</td>\n <td class='B3'></td>\n</tr>")
        self.assertEqual(te.eia_daily(html), [(date(1983, 4, 18), 30.75), (date(1983, 4, 19), 30.75),
                                              (date(1983, 4, 20), 30.70), (date(1983, 4, 21), 1030.68)])


class ExpiryTests(unittest.TestCase):
    def test_wti_expires_three_sessions_before_the_25th(self):
        sessions = _sessions(date(2026, 6, 1), date(2026, 12, 31), holidays={date(2026, 11, 26)})
        expiries = set(te.wti_expiries(sessions))
        self.assertIn(date(2026, 11, 20), expiries)   # the 25th is a Wednesday
        self.assertIn(date(2026, 7, 21), expiries)    # the 25th is a Saturday: count from Friday the 24th

    def test_gas_expires_three_sessions_before_the_delivery_month(self):
        sessions = _sessions(date(2026, 10, 1), date(2026, 12, 31), holidays={date(2026, 11, 26)})
        self.assertIn(date(2026, 11, 25), te.ng_expiries(sessions))   # Thanksgiving is not a session

    def test_no_expiry_is_invented_past_the_end_of_the_data(self):
        sessions = _sessions(date(2026, 10, 1), date(2026, 11, 13))
        self.assertTrue(all(e < date(2026, 11, 1) for e in te.wti_expiries(sessions)))
        self.assertTrue(all(e < date(2026, 11, 1) for e in te.ng_expiries(sessions)))


class RollTests(unittest.TestCase):
    def test_rolling_never_books_the_gap_between_contracts(self):
        # Every contract's price is constant and each later one is 2% dearer (contango): the contract
        # held never moves, so a correct roll shows no return at all.
        sessions = _sessions(date(2025, 1, 1), date(2025, 12, 31))
        expiries = te.wti_expiries(sessions)
        price = lambda n: 50.0 * 1.02 ** n  # noqa: E731
        c1, c2 = [], []
        for d in sessions:
            n = sum(e < d for e in expiries)
            c1.append((d, price(n)))
            c2.append((d, price(n + 1)))
        days, rets, rolls = te.futures_returns(c1, c2, expiries)
        self.assertTrue(all(abs(r) < 1e-12 for r in rets))
        self.assertEqual(sum(rolls), len([e for e in expiries if e >= days[0] + timedelta(days=10)]))


class StrategyTests(unittest.TestCase):
    def _series(self, n=700, drift=0.003, seed=1):
        rng = random.Random(seed)
        days = _sessions(date(2020, 1, 1), date(2030, 1, 1))[:n]
        return days, [drift + rng.gauss(0, 0.01) for _ in range(n)]

    def test_a_steady_rise_is_held_long_at_about_the_volatility_target(self):
        days, rets = self._series()
        out = te.strategy(days, rets, [False] * len(days), cost=0.0)
        positions = [w for _, _, w in out[5:]]
        self.assertTrue(all(w > 0 for w in positions))
        self.assertAlmostEqual(sum(positions) / len(positions), 0.10 / (0.01 * math.sqrt(252)), delta=0.15)

    def test_positions_change_only_after_a_weeks_last_session(self):
        days, rets = self._series(seed=2)
        out = te.strategy(days, rets, [False] * len(days), cost=0.0)
        for (d0, _, w0), (d1, _, w1) in zip(out, out[1:]):
            if w0 != w1:
                self.assertNotEqual(d0.isocalendar()[:2], d1.isocalendar()[:2])

    def test_costs_are_charged_on_trades_and_rolls(self):
        days, rets = self._series(seed=3)
        rolls = [d.day <= 3 and d.month != (d - timedelta(days=5)).month for d in days]
        free = sum(p for _, p, _ in te.strategy(days, rets, [False] * len(days), cost=0.0))
        paid = sum(p for _, p, _ in te.strategy(days, rets, rolls, cost=0.001))
        self.assertLess(paid, free)

    def test_the_signal_uses_no_later_returns(self):
        days, rets = self._series(seed=4)
        full = te.strategy(days, rets, [False] * len(days), cost=0.0005)
        cut = te.strategy(days[:500], rets[:500], [False] * 500, cost=0.0005)
        self.assertEqual([round(w, 12) for _, _, w in full[:len(cut)]], [round(w, 12) for _, _, w in cut])


class PortfolioTests(unittest.TestCase):
    def test_markets_count_once_they_have_a_position(self):
        a = [(date(2020, 1, 1), 0.01, 1.0), (date(2020, 1, 2), 0.02, 1.0)]
        b = [(date(2020, 1, 2), -0.02, 1.0)]
        days, rets, all_in = te.portfolio({"a": a, "b": b}, ("a", "b"))
        self.assertEqual(all_in, date(2020, 1, 2))
        self.assertEqual(days, [date(2020, 1, 1), date(2020, 1, 2)])
        self.assertAlmostEqual(rets[0], 0.01)
        self.assertAlmostEqual(rets[1], 0.0)

    def test_a_market_leaves_once_its_data_ends(self):
        a = [(date(2020, 1, 1), 0.01, 1.0), (date(2020, 1, 2), 0.02, 1.0), (date(2020, 1, 3), 0.03, 1.0)]
        b = [(date(2020, 1, 1), -0.01, 1.0)]
        _, rets, _ = te.portfolio({"a": a, "b": b}, ("a", "b"))
        self.assertAlmostEqual(rets[0], 0.0)
        self.assertAlmostEqual(rets[2], 0.03)

    def test_fall_blocks_pick_the_stretches_when_the_index_fell(self):
        days = _sessions(date(2020, 1, 1), date(2020, 12, 31))
        closes = [(d, 100.0 * (0.997 ** k if k < 60 else 1.0)) for k, d in enumerate(days)]
        port = [0.001 if k < 60 else -0.0005 for k in range(len(days))]
        blocks = te.fall_blocks(closes, days, port, days[0], days[-1])
        result = te.block_test(blocks)
        self.assertEqual(result["fall_blocks"], 2)
        self.assertGreater(result["mean_in_falls"], 0)


if __name__ == "__main__":
    unittest.main()
