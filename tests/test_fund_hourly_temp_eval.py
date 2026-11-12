from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_hourly_temp_eval", ROOT / "scripts" / "fund_hourly_temp_eval.py")
ht = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ht  # dataclasses look their module up
_spec.loader.exec_module(ht)

NY = ht.CITIES[0]


def _h(day, hour):
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


def _event(hour, value, strikes=(60.99, 61.99), quotes=None):
    ticker = f"KXTEMPNYCHS-{hour:%y%b%d%H}".upper()
    markets = []
    for k in strikes:
        markets.append({"ticker": f"{ticker}-T{k}", "close_time": hour.isoformat().replace("+00:00", "Z"),
                        "rules_primary": "... as reported by Synoptic Data, is above ...", "strike_type": "greater",
                        "floor_strike": k, "expiration_value": f"{value:.2f}", "result": "yes" if value > k else "no"})
    return {"event_ticker": ticker, "markets": markets}


class TimingTests(unittest.TestCase):
    def test_the_run_is_the_newest_hourly_one_ready_by_the_decision(self):
        # Decision 14:10 UTC for 15:00; runs must be initialised by 12:10, so the 12z run.
        self.assertEqual(ht.run_for(_h(5, 15)), _h(5, 12))
        for hour in range(24):
            run = ht.run_for(_h(5, hour))
            self.assertLessEqual(run, _h(5, hour) - ht.DECISION_LEAD - ht.RUN_READY_AFTER)
            self.assertEqual(_h(5, hour) - run, timedelta(hours=3))
        self.assertEqual(ht.needed_files([_h(5, 15)]), [(_h(5, 12), 2), (_h(5, 12), 3)])

    def test_a_city_forecast_is_the_mean_over_all_its_stations(self):
        run = _h(5, 12)
        temps = {s: 60.0 + k for k, s in enumerate(NY.members)}
        got = ht.city_forecasts(NY, {(run, 3): temps, (run, 2): {"KLGA": 50.0}})
        self.assertEqual(got, {run: {_h(5, 15): 60.0 + 3.5}})  # a file missing a station gives nothing

    def test_the_centre_is_the_last_reading_plus_hrrrs_change(self):
        hour = _h(5, 15)
        runs = {_h(5, 12): {_h(5, 14): 60.0, _h(5, 15): 61.5}}
        values = {_h(5, 14): 58.2}
        self.assertAlmostEqual(ht.centres(values, runs, [hour])[hour], 59.7)
        self.assertEqual(ht.centres(values, runs, [hour], use_hrrr=False)[hour], 58.2)
        self.assertEqual(ht.centres({}, runs, [hour]), {})


class ErrorTests(unittest.TestCase):
    def test_errors_come_only_from_settled_hours_at_a_similar_local_time(self):
        hour = _h(20, 18)  # 14:00 in New York
        f, values = {}, {}
        for back in range(1, 24 * 25):
            h = hour - timedelta(hours=back)
            f[h], values[h] = 50.0, 51.0
        f[hour], values[hour] = 50.0, 99.0  # the hour itself is not known yet
        errors = ht.recent_errors(hour, NY.tz, values, f)
        self.assertTrue(all(e == 1.0 for e in errors))
        self.assertEqual(len(errors), 21 * 5)  # 21 days x 5 local hours; the hour itself is left out

    def test_the_chance_counts_errors_with_a_half_count_prior(self):
        errors = [-1.0, 0.0, 0.5, 2.0]
        self.assertAlmostEqual(ht.chance_above(60.0, errors, 60.25), (2 + 0.5) / 5)
        self.assertAlmostEqual(ht.chance_above(60.0, errors, 70.0), 0.5 / 5)


class QuoteTests(unittest.TestCase):
    def test_the_quote_is_the_last_candle_closed_by_the_decision(self):
        at = _h(5, 14) + timedelta(minutes=10)
        rows = [{"end_period_ts": int((at - timedelta(minutes=m)).timestamp()),
                 "yes_bid": {"close_dollars": f"{0.30 + m / 100:.2f}"}, "yes_ask": {"close_dollars": "0.50"}}
                for m in (8, 3, -1)]
        self.assertEqual(ht.quote_at(rows, at), (0.33, 0.5))
        self.assertEqual(ht.quote_at([], at), (None, None))

    def test_the_trader_rule_buys_the_side_with_ten_points(self):
        self.assertEqual(ht.trader_rule(0.62, 0.40, 0.50), ("YES", 0.5))
        side, price = ht.trader_rule(0.20, 0.32, 0.40)
        self.assertEqual(side, "NO")
        self.assertAlmostEqual(price, 0.68)
        self.assertIsNone(ht.trader_rule(0.55, 0.47, 0.50))


class DecideTests(unittest.TestCase):
    def test_a_sharper_model_trades_and_is_scored_against_the_mid(self):
        hours = [_h(1, 0) + timedelta(hours=k) for k in range(24 * 16)]
        truth = [60.5 if k % 2 else 61.5 for k in range(len(hours))]
        events = [_event(h, truth[k], strikes=(60.99,)) for k, h in enumerate(hours)]
        # HRRR knows the alternation exactly, so the centre is always right.
        runs = {}
        for k, h in enumerate(hours[1:], start=1):
            run = runs.setdefault(ht.run_for(h), {})
            run[h], run[hours[k - 1]] = truth[k], truth[k - 1]
        candles = {}
        for ev in events:
            at = ht.event_hour(ev) - ht.DECISION_LEAD
            # The market leans the right way, but less than the model does.
            candles[ev["event_ticker"]] = {m["ticker"]: [{
                "end_period_ts": int(at.timestamp()),
                "yes_bid": {"close_dollars": "0.70" if m["result"] == "yes" else "0.25"},
                "yes_ask": {"close_dollars": "0.75" if m["result"] == "yes" else "0.30"}}] for m in ev["markets"]}
        old = (ht.FIRST_HOUR, ht.LAST_HOUR)
        ht.FIRST_HOUR, ht.LAST_HOUR = hours[0], hours[-1]
        try:
            brackets, trades, trader, coverage = ht.decide(NY, events, runs, candles)
            report = ht.report({NY.series: {"brackets": brackets, "trades": trades, "trader": trader,
                                            "coverage": coverage, "no_hrrr": {"brackets": [], "trades": []}}})
        finally:
            ht.FIRST_HOUR, ht.LAST_HOUR = old
        self.assertGreater(coverage["forecast"], 0)
        self.assertGreaterEqual(len(trades), ht.MIN_TRADES)
        self.assertGreater(ht.mean_return(trades), 0)
        self.assertGreater(ht.brier_skill(brackets), 0)
        self.assertEqual(report["verdict"], "GO")
        self.assertTrue(trader)

if __name__ == "__main__":
    unittest.main()
