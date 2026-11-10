from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_weather_lock_eval", _ROOT / "scripts" / "fund_weather_lock_eval.py")
lock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lock)

NY = fw.BY_SERIES["KXHIGHNY"]
DAY = date(2026, 11, 8)


def _at(h, m=0, day=8):
    return datetime(2026, 11, day, h, m, tzinfo=timezone.utc)


class ReadingTests(unittest.TestCase):
    CSV = ("station,valid,tmpf\nNYC,2026-11-08 04:51,70.00\nNYC,2026-11-08 05:51,55.00\nNYC,2026-11-08 18:51,61.10\n"
           "NYC,2026-11-08 19:51,M\nNYC,2026-11-08 20:51,63.00\n")

    def test_the_running_high_starts_at_local_midnight_and_stops_at_the_decision(self):
        obs = lock.readings(self.CSV)
        self.assertEqual(len(obs), 4)  # the missing reading is skipped
        # 04:51 UTC is still 7 November in New York (EST, UTC-5): its 70F does not count.
        self.assertEqual(lock.running_high(obs, DAY, NY.tz, _at(20)), 61.1)
        self.assertEqual(lock.running_high(obs, DAY, NY.tz, _at(21)), 63.0)
        self.assertIsNone(lock.running_high(obs, DAY, NY.tz, _at(5, 0)))


class DecidedTests(unittest.TestCase):
    def test_a_range_is_decided_only_a_full_degree_past_its_edge(self):
        below_61 = fw.bracket_bounds({"strike_type": "less", "cap_strike": 61})  # pays on a high under 60.5
        self.assertIsNone(lock.decided(below_61, 61.4))
        self.assertEqual(lock.decided(below_61, 61.5), "NO")
        between = fw.bracket_bounds({"strike_type": "between", "floor_strike": 61, "cap_strike": 62})
        self.assertIsNone(lock.decided(between, 63.4))
        self.assertEqual(lock.decided(between, 63.5), "NO")
        above_68 = fw.bracket_bounds({"strike_type": "greater", "floor_strike": 68})  # pays from 68.5
        self.assertIsNone(lock.decided(above_68, 69.4))
        self.assertEqual(lock.decided(above_68, 69.5), "YES")
        self.assertIsNone(lock.decided((-math.inf, math.inf), 99.0))

    def test_only_a_price_that_leaves_a_gain_after_the_fee_is_bought(self):
        self.assertAlmostEqual(lock.trade("NO", 0.03, 0.05), 0.97)
        self.assertEqual(lock.trade("YES", 0.97, 0.98), 0.98)
        self.assertIsNone(lock.trade("NO", 0.0, 0.01))  # No would cost a dollar
        self.assertIsNone(lock.trade("YES", 0.99, 1.0))
        self.assertIsNone(lock.trade("YES", 0.9, None))


class TradeTests(unittest.TestCase):
    def _data(self, result, settled="63.00"):
        candles = {"KXHIGHNY-26NOV08-T61": [{"end_period_ts": int(_at(20).timestamp()),
                                             "yes_bid": {"close_dollars": "0.0300"},
                                             "yes_ask": {"close_dollars": "0.0500"}}]}
        return {"events": [{"event_ticker": "KXHIGHNY-26NOV08", "markets": [
                    {"ticker": "KXHIGHNY-26NOV08-T61", "strike_type": "less", "cap_strike": 61, "result": result,
                     "expiration_value": settled}]}],
                "obs": lock.readings(ReadingTests.CSV.replace("61.10", "62.00")),
                "candles": {"KXHIGHNY-26NOV08": candles}}

    def test_a_decided_no_that_settles_no_earns_the_rest_of_the_dollar(self):
        [t] = lock.trades_at(NY, self._data("no"), 20)
        self.assertEqual((t["side"], t["won"]), ("NO", True))
        self.assertAlmostEqual(t["return"], (1 - 0.97 - fw.kalshi_fee(0.97)) / 0.97)

    def test_a_settlement_that_contradicts_the_airport_loses_the_stake(self):
        [t] = lock.trades_at(NY, self._data("yes", "60.00"), 20)
        self.assertFalse(t["won"])
        self.assertAlmostEqual(t["return"], (0 - 0.97 - fw.kalshi_fee(0.97)) / 0.97)

    def test_the_verdict_needs_a_hundred_trades_and_an_interval_above_zero(self):
        win = {"won": True, "return": 0.01, "city": "KXHIGHNY", "side": "NO", "price": 0.98, "ticker": "T",
               "high_so_far": 63.0, "settled": 63.0}
        many = [{**win, "day": f"2026-10-{d % 28 + 1:02d}"} for d in range(150)]
        self.assertEqual(lock.report(many, {}, through=DAY)["verdict"], "GO")
        self.assertEqual(lock.report(many[:60], {}, through=DAY)["verdict"], "NO-GO")
        lost = many + [{**win, "won": False, "return": -1.0, "day": "2026-10-02"} for _ in range(3)]
        self.assertEqual(lock.report(lost, {}, through=DAY)["verdict"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
