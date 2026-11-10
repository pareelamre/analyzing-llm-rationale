from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import fund_orchestrator as orch  # noqa: E402
from analyzing_llm_rationale import fund_weather as fw  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_weather_eval", _ROOT / "scripts" / "fund_weather_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

NY = fw.BY_SERIES["KXHIGHNY"]


def _market(ticker, kind, floor=None, cap=None, bid=None, ask=None, result=None, value=None, sub=""):
    return {"ticker": ticker, "strike_type": kind, "floor_strike": floor, "cap_strike": cap, "yes_bid_dollars": bid,
            "yes_ask_dollars": ask, "result": result, "expiration_value": value, "yes_sub_title": sub}


def _hourly(days, highs_by_model, variable="temperature_2m"):
    """An Open-Meteo multi-model payload whose hourly values peak at each model's high."""
    times, cols = [], {f"{variable}_{m}": [] for m in highs_by_model}
    for day in days:
        for h in range(24):
            times.append(f"{day.isoformat()}T{h:02d}:00")
            for m, highs in highs_by_model.items():
                cols[f"{variable}_{m}"].append(highs[day] - abs(h - 15))
    return {"hourly": {"time": times, **cols}}


class MarketTests(unittest.TestCase):
    def test_brackets_pay_on_whole_degrees(self):
        self.assertEqual(fw.bracket_bounds(_market("T61", "less", cap=61)), (-math.inf, 60.5))
        self.assertEqual(fw.bracket_bounds(_market("T68", "greater", floor=68)), (68.5, math.inf))
        self.assertEqual(fw.bracket_bounds(_market("B61.5", "between", 61, 62)), (60.5, 62.5))
        self.assertIsNone(fw.bracket_bounds(_market("X", "custom")))

    def test_event_tickers_name_their_day(self):
        self.assertEqual(fw.event_day("KXHIGHNY-26NOV08"), date(2026, 11, 8))
        self.assertEqual(fw.event_day("KXHIGHTSATX-26SEP03"), date(2026, 9, 3))
        self.assertIsNone(fw.event_day("KXHIGHNY"))

    def test_observed_highs_come_from_settled_events_on_the_new_source(self):
        settled = [{"event_ticker": "KXHIGHNY-26NOV08", "markets": [_market("a", "less", cap=61, value="60.00"),
                                                                   _market("b", "between", 61, 62, value="60.00")]},
                   {"event_ticker": "KXHIGHNY-26AUG30", "markets": [_market("c", "less", cap=80, value="79.00")]}]
        self.assertEqual(fw.observed_highs(settled), {date(2026, 11, 8): 60.0})

    def test_every_city_has_a_station_and_time_zone(self):
        self.assertEqual(len(fw.CITIES), 24)
        for c in fw.CITIES:
            self.assertTrue(c.series.startswith("KXHIGH") and c.cli.startswith("CLI") and c.station.startswith("K"))
            self.assertTrue(-125 < c.lon < -70 and 25 < c.lat < 48, c)


class ForecastTests(unittest.TestCase):
    def test_each_models_high_is_its_hottest_local_hour_and_short_days_are_dropped(self):
        day = date(2026, 11, 10)
        payload = _hourly([day], {"gfs_seamless": {day: 60.0}, "ecmwf_ifs025": {day: 62.0}})
        payload["hourly"]["time"] = payload["hourly"]["time"] + ["2026-11-11T00:00"]
        for k in ("temperature_2m_gfs_seamless", "temperature_2m_ecmwf_ifs025"):
            payload["hourly"][k] = payload["hourly"][k] + [99.0]
        highs = fw.daily_highs(payload)
        self.assertEqual(highs["gfs_seamless"], {day: 60.0})  # one hour of the 11th is not a day
        self.assertEqual(highs["ecmwf_ifs025"][day], 62.0)

    def test_the_ensemble_needs_three_models(self):
        day = date(2026, 11, 10)
        self.assertIsNone(fw.ensemble({"a": {day: 60}, "b": {day: 62}}, day))
        mean, spread, n = fw.ensemble({"a": {day: 60}, "b": {day: 62}, "c": {day: 64}}, day)
        self.assertEqual((mean, spread, n), (62.0, 2.0, 3))


class CalibrationTests(unittest.TestCase):
    def _rows(self, n=60, bias=2.0):
        rows = []
        start = date(2026, 9, 3)
        for i in range(n):
            spread = 0.5 + (i % 5)
            error = (1.0 if i % 2 else -1.0) * math.sqrt(1.0 + spread ** 2)
            rows.append(fw.Row("KXHIGHNY", start + timedelta(days=i), 60.0, spread, 60.0 + bias + error))
        return rows

    def test_bias_and_spread_come_only_from_days_already_known(self):
        rows = self._rows()
        self.assertIsNone(fw.fit(rows, date(2026, 9, 15)))  # under 21 days of history
        cal = fw.fit(rows, date(2026, 10, 31))
        self.assertAlmostEqual(cal.bias["KXHIGHNY"], 2.0, delta=0.4)
        self.assertGreater(cal.d, 0.5)  # more disagreement, more spread
        self.assertGreater(cal.sigma(4.5), cal.sigma(0.5))
        # A day after the cut-off changes nothing.
        later = rows + [fw.Row("KXHIGHNY", date(2026, 11, 1), 60.0, 1.0, 90.0)]
        self.assertEqual(fw.fit(later, date(2026, 10, 31)).bias, cal.bias)

    def test_sigma_never_falls_below_a_degree(self):
        cal = fw.Calibration({}, 0.0, 0.0, 30, date(2026, 11, 1))
        self.assertEqual(cal.sigma(0.0), fw.MIN_SIGMA)

    def test_chances_over_a_days_brackets_add_to_one(self):
        bounds = [(-math.inf, 60.5), (60.5, 62.5), (62.5, 64.5), (64.5, math.inf)]
        self.assertAlmostEqual(sum(fw.chance(62.0, 2.0, b) for b in bounds), 1.0, places=9)


class RuleTests(unittest.TestCase):
    def test_kalshis_fee_comes_off_the_edge(self):
        side, price, edge = fw.rule(0.80, 0.60, 0.62)
        self.assertEqual((side, price), ("YES", 0.62))
        self.assertAlmostEqual(edge, (0.80 - 0.62 - 0.07 * 0.62 * 0.38) / 0.62)

    def test_no_is_bought_at_one_minus_the_bid(self):
        side, price, _ = fw.rule(0.05, 0.30, 0.32)
        self.assertEqual(side, "NO")
        self.assertAlmostEqual(price, 0.70)

    def test_small_gaps_far_prices_and_likely_errors_are_not_traded(self):
        self.assertIsNone(fw.rule(0.66, 0.60, 0.62))  # 4 points
        self.assertIsNone(fw.rule(0.10, 0.01, 0.02))  # 2c is outside the band, and so is 99c for NO
        self.assertIsNone(fw.rule(0.95, 0.40, 0.42))  # 54 points from the mid
        self.assertIsNone(fw.rule(0.5, None, 0.42))

    def test_a_day_is_priced_from_forecasts_only_before_its_morning(self):
        day = date(2026, 11, 10)
        self.assertTrue(fw.tradeable_day(day, NY, datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)))  # 03:00 ET
        self.assertFalse(fw.tradeable_day(day, NY, datetime(2026, 11, 10, 20, 0, tzinfo=timezone.utc)))
        self.assertTrue(fw.tradeable_day(day, NY, datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)))


class SignalTests(unittest.TestCase):
    NOW = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)

    def _http(self, forecast_high=66.0):
        day = date(2026, 11, 10)
        history_days = [day - timedelta(days=i) for i in range(1, 50)]
        models = {m: {d: 60.0 + (i % 3) for d in history_days} for i, m in enumerate(fw.MODELS)}
        settled = [{"event_ticker": f"KXHIGHNY-26{d:%b}{d:%d}".upper(),
                    "markets": [_market("x", "less", cap=70, value=f"{61.0 + (j % 2) * 2 - 1:.2f}")]}
                   for j, d in enumerate(history_days)]
        open_ev = [{"event_ticker": "KXHIGHNY-26NOV10", "markets": [
            _market("KXHIGHNY-26NOV10-T61", "less", cap=61, bid=0.20, ask=0.22, sub="60° or below"),
            _market("KXHIGHNY-26NOV10-B65.5", "between", 65, 66, bid=0.20, ask=0.22, sub="65° to 66°"),
            _market("KXHIGHNY-26NOV10-T66", "greater", floor=66, bid=0.20, ask=0.22, sub="67° or above")]}]
        live = _hourly([day, day + timedelta(days=1)], {m: {day: forecast_high, day + timedelta(days=1): 50.0}
                                                        for m in fw.MODELS})

        def get(url, params, timeout, headers):
            if url.endswith("/events"):
                if params["series_ticker"] != "KXHIGHNY":
                    body = {"events": []}
                else:
                    body = {"events": open_ev if params["status"] == "open" else settled}
            elif url == fw.PREVIOUS_RUNS_API:
                body = _hourly(history_days, models, "temperature_2m_previous_day1")
            else:
                body = live
            return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)
        return get

    def test_a_forecast_well_away_from_the_prices_trades_with_its_reasons(self):
        out = fw.weather_signals(self.NOW, http_get=self._http())
        sides = {s.instrument["id"]: s.side for s in out["signals"]}
        self.assertEqual(sides.get("kalshi:KXHIGHNY-26NOV10-T61"), "NO")  # a 66F day is rarely 60F or below
        self.assertTrue(all(s.analyst == "weather_model" for s in out["signals"]))
        s = out["signals"][0]
        self.assertIn("Kalshi", {f["source"] for f in s.facts})
        self.assertEqual(len(out["priced"]), 3)
        # One city-day's brackets together stay inside the event cap.
        total = sum(orch.KELLY_FRACTION * s.trust * s.kelly for s in out["signals"])
        self.assertLessEqual(total, fw.EVENT_CAP + 1e-9)

    def test_a_market_whose_morning_has_passed_is_left_alone(self):
        later = datetime(2026, 11, 10, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(fw.weather_signals(later, http_get=self._http())["signals"], [])


class EvalTests(unittest.TestCase):
    def test_a_quote_is_the_last_candle_closed_by_the_decision_and_not_too_old(self):
        at = datetime(2026, 11, 8, 8, 0, tzinfo=timezone.utc)
        rows = [{"end_period_ts": int(at.timestamp()) - 7200, "yes_bid": {"close_dollars": "0.40"}, "yes_ask": {"close_dollars": "0.42"}},
                {"end_period_ts": int(at.timestamp()), "yes_bid": {"close_dollars": "0.50"}, "yes_ask": {"close_dollars": "0.52"}},
                {"end_period_ts": int(at.timestamp()) + 3600, "yes_bid": {"close_dollars": "0.90"}, "yes_ask": {"close_dollars": "0.92"}}]
        self.assertEqual(ev.quote_at(rows, at), (0.50, 0.52))
        self.assertEqual(ev.quote_at(rows[:1], at + timedelta(hours=2)), (None, None))  # over 3 hours old

    def test_the_bootstrap_resamples_whole_days(self):
        trades = [{"day": f"2026-10-{d:02d}", "return": 0.1 if d % 2 else -0.05} for d in range(1, 31) for _ in range(3)]
        lo, hi = ev.bootstrap(trades, ev.mean_return)
        self.assertLess(lo, ev.mean_return(trades))
        self.assertGreater(hi, ev.mean_return(trades))

    def test_the_verdict_is_go_only_when_the_whole_interval_is_above_zero(self):
        good = [{"day": f"2026-10-{d:02d}", "decision": "A", "city": "KXHIGHNY", "return": 0.2}
                for d in range(1, 29)]
        self.assertEqual(ev.report([], good, through=date(2026, 11, 9))["verdict"], "GO")
        mixed = [{**t, "return": 0.2 if i % 2 else -0.19} for i, t in enumerate(good)]
        self.assertEqual(ev.report([], mixed, through=date(2026, 11, 9))["verdict"], "SHADOW")


if __name__ == "__main__":
    unittest.main()
