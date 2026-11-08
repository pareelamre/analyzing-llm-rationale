from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import index_forecast as ixf  # noqa: E402

_spec = importlib.util.spec_from_file_location("index_forecast_tick", _ROOT / "scripts" / "index_forecast_tick.py")
tick = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tick)

IST = ixf.IST


def _sessions(start: date, n: int, holidays=()) -> list[date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5 and day not in holidays:
            out.append(day)
        day += timedelta(days=1)
    return out


def _series(n=1100, vol=0.01, seed=0, start=date(2022, 1, 3)):
    rng = np.random.default_rng(seed)
    days = _sessions(start, n)
    logp = np.log(20000) + np.cumsum(rng.normal(0, vol, n))
    return list(zip(days, np.exp(logp).tolist()))


def _vix(days, level=16.0):
    return [(d, level) for d in days]


class FitTests(unittest.TestCase):
    def test_a_steady_series_gets_its_own_volatility_back(self):
        closes = _series(vol=0.01)
        fit = ixf.fit_horizon(closes, _vix([d for d, _ in closes], 0.01 * math.sqrt(252) * 100), 1)

        self.assertAlmostEqual(fit["sigma"], 0.01, delta=0.0015)
        self.assertAlmostEqual(fit["baseline_sigma"], 0.01, delta=0.003)
        self.assertEqual(fit["as_of"], closes[-1][0])

    def test_the_next_session_trains_on_recent_history_and_longer_horizons_on_all_of_it(self):
        closes = _series(n=2000)
        vix = _vix([d for d, _ in closes])
        # Rows with a VIX reading, past the warm-up, whose target session has closed.
        usable = {h: len(closes) - 1 - ixf.WARMUP_SESSIONS - h for h in ixf.HORIZONS}

        self.assertEqual(len(ixf.fit_horizon(closes, vix, 1)["z"]), 750)
        self.assertEqual(len(ixf.fit_horizon(closes, vix, 5)["z"]), usable[5])
        self.assertEqual(len(ixf.fit_horizon(closes, vix, 21)["z"]), usable[21])

    def test_quantiles_rise_and_level_chances_fall(self):
        closes = _series()
        forecast = ixf.forecast_index("Nifty 50", closes, _vix([d for d, _ in closes]), [])

        self.assertEqual([hz["h"] for hz in forecast["horizons"]], [1, 5, 21])
        for hz in forecast["horizons"]:
            self.assertEqual(hz["quantiles"], sorted(hz["quantiles"]))
            chances = [lvl["p_above"] for lvl in hz["levels"]]
            self.assertEqual(chances, sorted(chances, reverse=True))
            self.assertTrue(0.3 < hz["p_up"] < 0.7)
        widths = [hz["quantiles"][5] - hz["quantiles"][1] for hz in forecast["horizons"]]
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])

    def test_short_history_gives_no_forecast(self):
        closes = _series(n=400)
        self.assertIsNone(ixf.forecast_index("Nifty 50", closes, _vix([d for d, _ in closes]), []))

    def test_target_dates_skip_weekends_and_holidays(self):
        friday = date(2026, 11, 6)
        self.assertEqual(ixf.next_sessions(friday, 2, [date(2026, 11, 9)]), [date(2026, 11, 10), date(2026, 11, 11)])

    def test_round_levels_bracket_the_close_and_stay_positive(self):
        levels = ixf.round_levels(25030.35, 0.009)
        # sigma 0.9% of 25,030 is 226 points; half of that is nearest a 100-point step.
        self.assertEqual(levels, [24800, 24900, 25000, 25100, 25200, 25300])
        self.assertTrue(all(level > 0 for level in ixf.round_levels(4000, 0.6)))


class IssueWindowTests(unittest.TestCase):
    def test_open_from_the_close_until_the_next_session_opens(self):
        friday = date(2026, 11, 6)
        at = lambda d, h, m: datetime(d.year, d.month, d.day, h, m, tzinfo=IST)  # noqa: E731

        self.assertFalse(ixf.issue_window_open(friday, at(friday, 15, 0), []))
        self.assertTrue(ixf.issue_window_open(friday, at(friday, 19, 0), []))
        self.assertTrue(ixf.issue_window_open(friday, at(date(2026, 11, 9), 9, 0), []))
        self.assertFalse(ixf.issue_window_open(friday, at(date(2026, 11, 9), 9, 15), []))
        # Monday a holiday: the window runs on to Tuesday's open.
        self.assertTrue(ixf.issue_window_open(friday, at(date(2026, 11, 9), 12, 0), [date(2026, 11, 9)]))


class LedgerTests(unittest.TestCase):
    def _data(self, closes):
        return {"levels": {"Nifty 50": closes}, "vix": _vix([d for d, _ in closes]), "holidays": []}

    def test_run_records_once_and_scores_when_the_target_closes(self):
        closes = _series(n=1100)
        as_of = closes[-6][0]
        evening = datetime.combine(as_of, datetime.min.time(), tzinfo=IST) + timedelta(hours=19)

        payload, ledger = ixf.run(self._data(closes[:-5]), [], evening)
        self.assertEqual(len(ledger), 3)
        self.assertEqual({row["as_of"] for row in ledger}, {as_of.isoformat()})
        self.assertEqual(payload["track_record"]["pending"], 3)
        self.assertNotIn("recorded", payload["indices"][0])

        # A rerun the same evening adds nothing.
        _, ledger = ixf.run(self._data(closes[:-5]), ledger, evening)
        self.assertEqual(len(ledger), 3)

        # Five sessions later the 1- and 5-session rows score against those closes.
        later = datetime.combine(closes[-1][0], datetime.min.time(), tzinfo=IST) + timedelta(hours=19)
        payload, ledger = ixf.run(self._data(closes), ledger, later)
        by_h = {row["h"]: row for row in ledger if row["as_of"] == as_of.isoformat()}
        self.assertEqual(by_h[1]["realized"], round(closes[-5][1], 2))
        self.assertEqual(by_h[1]["realized_date"], closes[-5][0].isoformat())
        self.assertEqual(by_h[5]["realized"], round(closes[-1][1], 2))
        self.assertIsNone(by_h[21].get("realized"))
        summary = {row["h"]: row for row in payload["track_record"]["by_horizon"]}
        self.assertEqual(summary[1]["n"], 1)
        self.assertEqual(summary[21]["n"], 0)
        self.assertEqual(len(payload["track_record"]["recent"]), 2)

    def test_a_forecast_built_after_the_next_open_is_kept_out_of_the_ledger(self):
        closes = _series(n=1100)
        late = datetime.combine(closes[-1][0] + timedelta(days=3), datetime.min.time(), tzinfo=IST) + timedelta(hours=12)

        payload, ledger = ixf.run(self._data(closes), [], late)

        self.assertEqual(ledger, [])
        self.assertIs(payload["indices"][0]["recorded"], False)

    def test_the_month_ahead_shape_pools_every_index_and_shorter_horizons_keep_their_own(self):
        calm = _series(seed=1)
        # A second index with fat tails: a 5% jump every 40 sessions.
        rng = np.random.default_rng(2)
        jumps = np.where(np.arange(len(calm)) % 40 == 0, rng.choice([-0.05, 0.05], len(calm)), 0.0)
        wild = list(zip([d for d, _ in calm], np.exp(np.log(20000) + np.cumsum(rng.normal(0, 0.01, len(calm)) + jumps)).tolist()))
        data = {"levels": {"Nifty 50": calm, "Nifty Bank": wild}, "vix": _vix([d for d, _ in calm]), "holidays": []}
        evening = datetime.combine(calm[-1][0], datetime.min.time(), tzinfo=IST) + timedelta(hours=19)

        payload, _ = ixf.run(data, [], evening)
        in_run = {hz["h"]: hz for hz in next(f for f in payload["indices"] if f["name"] == "Nifty 50")["horizons"]}
        alone = {hz["h"]: hz for hz in ixf.forecast_index("Nifty 50", calm, data["vix"], [])["horizons"]}
        fits = {name: ixf.fit_index(closes, data["vix"]) for name, closes in data["levels"].items()}
        pooled = {hz["h"]: hz for hz in ixf.build_forecast("Nifty 50", fits["Nifty 50"], [], ixf.pooled_shapes(fits.values()))["horizons"]}

        self.assertEqual(in_run[1]["quantiles"], alone[1]["quantiles"])
        self.assertEqual(in_run[5]["quantiles"], alone[5]["quantiles"])
        self.assertNotEqual(in_run[21]["quantiles"], alone[21]["quantiles"])
        self.assertEqual(in_run[21]["quantiles"], pooled[21]["quantiles"])
        self.assertEqual(payload["model"]["pooled_shape_horizons"], [21])

    def test_an_index_without_history_is_left_out_with_a_note(self):
        closes = _series(n=1100)
        evening = datetime.combine(closes[-1][0], datetime.min.time(), tzinfo=IST) + timedelta(hours=19)

        payload, _ = ixf.run(self._data(closes), [], evening)

        self.assertEqual([item["name"] for item in payload["indices"]], ["Nifty 50"])
        self.assertTrue(any("Nifty Bank" in note for note in payload["notes"]))

    def test_coverage_and_skill_come_from_the_scored_rows(self):
        row = {"name": "Nifty 50", "as_of": "2026-11-02", "h": 1, "close": 100.0, "p_up": 0.5,
               "quantiles": [95, 96, 98, 100, 102, 104, 105], "baseline_quantiles": [90, 92, 96, 100, 104, 108, 110]}
        series = [(date(2026, 11, 2), 100.0), (date(2026, 11, 3), 103.0)]

        self.assertEqual(ixf.score_ledger([row], {"Nifty 50": series}), 1)
        self.assertEqual((row["in_50"], row["in_80"], row["in_90"], row["up"]), (False, True, True, True))
        summary = ixf.summarize_ledger([row])[0]
        self.assertEqual(summary["coverage_80"], 1.0)
        self.assertGreater(summary["skill_vs_baseline"], 0)


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class UpstoxFetchTests(unittest.TestCase):
    def test_a_run_fetches_full_history_in_decade_chunks_without_a_token(self):
        calls = []

        def get(url, timeout, headers):
            calls.append(url)
            self.assertNotIn("Authorization", headers)
            self.assertIn("Foresea", headers["User-Agent"])
            if "/market/holidays" in url:
                rows = [
                    {"date": "2026-11-24", "holiday_type": "TRADING_HOLIDAY", "closed_exchanges": ["NSE", "BSE"]},
                    {"date": "2026-11-08", "holiday_type": "SPECIAL_TIMING", "closed_exchanges": []},
                    {"date": "2026-08-26", "holiday_type": "SETTLEMENT_HOLIDAY", "closed_exchanges": []},
                    {"date": "2026-11-19", "holiday_type": "TRADING_HOLIDAY", "closed_exchanges": ["MCX"]},
                ]
                if url.endswith("/holidays/2027-01-01"):
                    rows = [{"date": "2027-01-01", "holiday_type": "TRADING_HOLIDAY", "closed_exchanges": ["NSE"]}]
                elif not url.endswith("/holidays"):
                    rows = []
                return _Resp(200, {"status": "success", "data": rows})
            if "/intraday/" in url:
                # Monday's candle from the intraday route; the historical route has not caught up.
                return _Resp(200, {"status": "success", "data": {"candles": [["2026-11-09T00:00:00+05:30", 1, 1, 1, 102.0, 0, 0]]}})
            if url.endswith("/2026-11-09/2016-12-31"):
                return _Resp(200, {"status": "success", "data": {"candles": [
                    ["2026-11-08T00:00:00+05:30", 1, 1, 1, 999.0, 0, 0],  # Sunday Muhurat session
                    ["2026-11-06T00:00:00+05:30", 1, 1, 1, 101.0, 0, 0],
                ]}})
            return _Resp(200, {"status": "success", "data": {"candles": [["2008-01-01T00:00:00+05:30", 1, 1, 1, 50.0, 0, 0]]}})

        monday_evening = datetime(2026, 11, 9, 13, 10, tzinfo=timezone.utc)  # 18:40 IST
        data = ixf.fetch_market_data(monday_evening, http_get=get)

        # Per series: today's candle plus two decade chunks back to 2008. Then the holiday list
        # and each weekday of next year inside the 60-day horizon (1-8 January 2027).
        self.assertEqual(len(calls), 11 * 3 + 1 + 6)
        self.assertIn("https://api.upstox.com/v3/historical-candle/NSE_INDEX%7CNifty%20Fin%20Service/days/1/2026-11-09/2016-12-31", calls)
        self.assertIn("https://api.upstox.com/v3/historical-candle/NSE_INDEX%7CIndia%20VIX/days/1/2016-12-30/2008-01-01", calls)
        expected = [(date(2008, 1, 1), 50.0), (date(2026, 11, 6), 101.0), (date(2026, 11, 9), 102.0)]
        self.assertEqual(data["levels"]["Nifty 50"], expected)
        self.assertEqual(data["vix"], expected)
        self.assertEqual(data["holidays"], [date(2026, 11, 24), date(2027, 1, 1)])

    def test_a_candle_is_not_used_before_its_close_settles(self):
        def get(url, timeout, headers):
            if "/market/holidays" in url:
                return _Resp(200, {"status": "success", "data": []})
            # A pre-open candle for Tuesday from the intraday route.
            return _Resp(200, {"status": "success", "data": {"candles": [
                ["2026-11-10T00:00:00+05:30", 1, 1, 1, 103.0, 0, 0], ["2026-11-09T00:00:00+05:30", 1, 1, 1, 102.0, 0, 0],
            ]}})

        tuesday_morning = datetime(2026, 11, 10, 3, 37, tzinfo=timezone.utc)  # 09:07 IST
        data = ixf.fetch_market_data(tuesday_morning, http_get=get)
        self.assertEqual(data["levels"]["Nifty 50"], [(date(2026, 11, 9), 102.0)])

    def test_a_failed_index_is_left_empty_but_a_failed_vix_raises(self):
        def get(url, timeout, headers):
            if "Nifty%20IT" in url or "India%20VIX" in url:
                return _Resp(429, {"status": "error"})
            if "/market/holidays" in url:
                return _Resp(200, {"status": "success", "data": []})
            return _Resp(200, {"status": "success", "data": {"candles": []}})

        with self.assertRaisesRegex(ixf.UpstoxError, "India%20VIX.*429"):
            ixf.fetch_market_data(datetime(2026, 11, 9, 13, 10, tzinfo=timezone.utc), http_get=get)


class TickScheduleTests(unittest.TestCase):
    def test_skips_once_the_latest_close_is_published(self):
        saturday = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
        previous = {"as_of": "2026-11-06", "generated_at": "2026-11-06T13:10:00+00:00", "holidays": []}
        self.assertTrue(tick.already_published(previous, saturday))

    def test_fetches_on_every_run_until_the_new_close_is_published(self):
        monday_evening = datetime(2026, 11, 9, 13, 10, tzinfo=timezone.utc)  # 18:40 IST
        previous = {"as_of": "2026-11-06", "generated_at": "2026-11-06T13:10:00+00:00", "holidays": []}
        self.assertFalse(tick.already_published(previous, monday_evening))
        retried = {**previous, "generated_at": "2026-11-09T12:40:00+00:00"}
        self.assertFalse(tick.already_published(retried, monday_evening))
        self.assertTrue(tick.already_published({**retried, "as_of": "2026-11-09"}, monday_evening))

    def test_monday_morning_still_counts_friday_as_the_latest_close(self):
        monday_morning = datetime(2026, 11, 9, 2, 0, tzinfo=timezone.utc)  # 07:30 IST
        self.assertEqual(tick._latest_close(monday_morning, []), date(2026, 11, 6))
        self.assertEqual(tick._latest_close(monday_morning + timedelta(hours=9), []), date(2026, 11, 9))


if __name__ == "__main__":
    unittest.main()
