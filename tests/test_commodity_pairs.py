from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import commodity_pairs as cp  # noqa: E402

_spec = importlib.util.spec_from_file_location("commodity_pairs_tick", _ROOT / "scripts" / "commodity_pairs_tick.py")
tick = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tick)

GOLD_SILVER = cp.PAIRS[0]


def _weekdays(start: date, n: int) -> list[date]:
    out, day = [], start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def _series(days, gold, silver):
    return {("gold", GOLD_SILVER["source"]): list(zip(days, map(float, gold))),
            ("silver", GOLD_SILVER["source"]): list(zip(days, map(float, silver)))}


def _cointegrated(n=200, seed=1, shock_at=None):
    """Silver a random walk; gold tied to it with a mean-reverting gap, plus an optional late shock."""
    rng = np.random.default_rng(seed)
    ls = np.log(30) + np.cumsum(rng.normal(0, 0.015, n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.8 * gap[t - 1] + rng.normal(0, 0.01)
    if shock_at is not None:
        gap[shock_at:] += 0.08
    lg = np.log(2400) + 0.9 * (ls - ls[0]) + gap
    return np.exp(lg), np.exp(ls)


class _Resp:
    def __init__(self, status, body=None, text="", content=b""):
        self.status_code, self._body, self.text, self.content = status, body, text, content

    def json(self):
        return self._body


def _contract(symbol, expiry):
    return {"key": f"MCX_FO|{symbol}{expiry:%m%d}", "contract": f"{symbol} FUT {expiry:%d %b %y}".upper(), "expiry": expiry}


def _listed(symbol, key, expiry):
    return {"instrument_type": "FUT", "underlying_symbol": symbol, "instrument_key": key, "trading_symbol": f"{symbol} FUT",
            "expiry": int(datetime.combine(expiry, datetime.min.time(), tzinfo=cp.IST).timestamp() * 1000) + 86_399_000}


class FeedTests(unittest.TestCase):
    def test_fred_skips_blank_holidays_weekends_and_today(self):
        csv = "observation_date,X\n2026-11-02,107.39\n2026-11-03,\n2026-11-07,99.0\n2026-11-09,101.5\n"
        seen = []

        def get(url, params, timeout, headers):
            seen.append(params["id"])
            self.assertEqual(params["cosd"], (date(2026, 11, 9) - timedelta(days=cp.FRED_DAYS)).isoformat())
            self.assertIn("Foresea", headers["User-Agent"])
            return _Resp(200, text=csv)

        series = cp.fetch_fred(date(2026, 11, 9), http_get=get)

        self.assertEqual(sorted(seen), sorted(cp.FRED_SERIES.values()))
        self.assertEqual(series[("brentcrude", "international")], [(date(2026, 11, 2), 107.39)])

    def test_the_front_rolls_before_expiry_and_never_steps_back(self):
        nov, dec = _contract("COPPER", date(2026, 11, 30)), _contract("COPPER", date(2026, 12, 31))
        candles = {
            nov["contract"]: {date(2026, 11, 20): (100.0, 5), date(2026, 11, 23): (101.0, 5), date(2026, 11, 24): (101.5, 0),
                              date(2026, 11, 25): (102.0, 5)},
            # No December trades on the 20th or the 25th; a candle without volume on the 24th.
            dec["contract"]: {date(2026, 11, 20): (102.0, 0), date(2026, 11, 23): (103.0, 5), date(2026, 11, 24): (103.5, 0)},
        }
        rows = cp.front_rows("copper", [nov, dec], candles, date(2026, 11, 20), date(2026, 11, 25))

        # 20th: November, ten days out. 23rd: seven days out, so December. 24th: nothing traded.
        # 25th: only November traded, and the series does not step back to it.
        self.assertEqual([(r["date"], r["contract"], r["close"]) for r in rows],
                         [("2026-11-20", nov["contract"], 100.0), ("2026-11-23", dec["contract"], 103.0)])
        self.assertEqual((rows[0]["next_contract"], rows[0]["next_close"]), (dec["contract"], 102.0))

    def test_a_roll_rescales_earlier_prices_and_keeps_every_daily_move(self):
        rows = [
            {"date": "2026-11-19", "contract": "NOV", "close": 100.0, "next_contract": "DEC", "next_close": 102.0},
            {"date": "2026-11-20", "contract": "NOV", "close": 101.0, "next_contract": "DEC", "next_close": 103.0},
            {"date": "2026-11-23", "contract": "DEC", "close": 104.0},
        ]
        series = cp.continuous(rows)

        self.assertEqual(series[-1], (date(2026, 11, 23), 104.0))
        prices = [price for _, price in series]
        self.assertAlmostEqual(prices[1] / prices[0], 101 / 100)
        self.assertAlmostEqual(prices[2] / prices[1], 104 / 103)

    def test_mcx_backfills_once_then_appends_after_the_last_stored_session(self):
        listed = [_listed("COPPER", "MCX_FO|1", date(2026, 11, 30)), _listed("COPPERM", "MCX_FO|2", date(2026, 11, 30)),
                  _listed("ZINC", "MCX_FO|3", date(2026, 11, 30))]
        calls = []

        def get(url, params, timeout, headers):
            calls.append(url)
            if url == cp.MCX_INSTRUMENTS_URL:
                return _Resp(200, content=gzip.compress(json.dumps(listed).encode()))
            if "MCX_FO%7C3" in url:
                return _Resp(500)
            frm = date.fromisoformat(url.rsplit("/", 1)[-1])
            candles = [[f"{d.isoformat()}T00:00:00+05:30", 0, 0, 0, 100.0 + d.day, 10, 0]
                       for d in (date(2026, 11, 5), date(2026, 11, 6)) if d >= frm]
            return _Resp(200, {"status": "success", "data": {"candles": candles}})

        # Zinc's candles fail to load: it is skipped, and the mini copper contract is never fetched.
        first = cp.fetch_mcx([], date(2026, 11, 7), http_get=get)
        self.assertEqual([(r["commodity"], r["date"], r["close"], r["expiry"]) for r in first],
                         [("copper", "2026-11-05", 105.0, "2026-11-30"), ("copper", "2026-11-06", 106.0, "2026-11-30")])
        self.assertNotIn("MCX_FO%7C2", "".join(calls))

        calls.clear()
        later = cp.fetch_mcx(first[:1], date(2026, 11, 9), http_get=get)
        self.assertTrue(any(c.endswith("MCX_FO%7C1/days/1/2026-11-08/2026-11-06") for c in calls))
        self.assertEqual([r["date"] for r in later], ["2026-11-06"])

    def test_gold_is_put_per_kg_alongside_the_fred_series(self):
        stored = [{"date": "2026-11-06", "commodity": "gold", "contract": "GOLD", "close": 142167.0}]
        fred = {("crudeoil", "international"): [(date(2026, 11, 3), 99.64)]}
        series = cp.build_series(stored, fred)

        self.assertEqual(series[("gold", "india")], [(date(2026, 11, 6), 14216700.0)])
        self.assertEqual(series[("crudeoil", "international")], fred[("crudeoil", "international")])

    def test_pairs_without_a_feed_are_paused(self):
        for key in ("gold-platinum", "gold-parity"):
            row = cp.track_pair(next(p for p in cp.PAIRS if p["key"] == key), {})
            self.assertTrue(row["paused"])
            self.assertNotIn("z", row)

    def test_parity_pairs_the_exchange_price_with_the_landed_cost(self):
        days = _weekdays(date(2026, 4, 6), 3)
        series = {("gold", "india"): [(d, 350.0 + i) for i, d in enumerate(days)],
                  ("gold", "landed"): [(d, 320.0 + i) for i, d in enumerate(days)]}
        _, la, lb = cp.aligned(cp.PAIRS[-1], series)

        np.testing.assert_allclose(np.exp(la - lb), [350 / 320, 351 / 321, 352 / 322])


class TrackerTests(unittest.TestCase):
    def test_a_cointegrated_pair_reads_cointegrated_with_a_short_half_life(self):
        days = _weekdays(date(2025, 6, 2), 320)
        gold, silver = _cointegrated(320)
        row = cp.track_pair(GOLD_SILVER, _series(days, gold, silver))

        self.assertEqual(row["hedge"], "regression")
        self.assertAlmostEqual(row["beta"], 0.9, delta=0.15)
        self.assertTrue(row["cointegrated"])
        self.assertLess(row["half_life"], 10)
        self.assertFalse(row["short_history"])
        self.assertEqual(len(row["z_recent"]), cp.Z_WINDOW)

    def test_a_shock_to_the_gap_makes_the_pair_stretched_with_the_rich_leg_named(self):
        days = _weekdays(date(2025, 6, 2), 200)
        gold, silver = _cointegrated(200, shock_at=199)
        row = cp.track_pair(GOLD_SILVER, _series(days, gold, silver))

        self.assertGreater(row["z"], cp.ENTRY_Z)
        self.assertEqual((row["stretched"], row["rich"], row["cheap"]), (True, "Gold", "Silver"))

    def test_the_move_back_to_usual_closes_the_gap_in_either_leg(self):
        days = _weekdays(date(2025, 6, 2), 200)
        gold, silver = _cointegrated(200, shock_at=199)
        row = cp.track_pair(GOLD_SILVER, _series(days, gold, silver))
        last, beta = row["_hist"][-1], row["_hist"][-1]["beta"]
        la, lb = row["_la"][-1], row["_lb"][-1]

        self.assertEqual(row["legs"], ["Gold", "Silver"])
        # Gold is rich, so gold falls or silver rises to get back to the window average.
        self.assertLess(row["to_usual"]["a"], 0)
        self.assertGreater(row["to_usual"]["b"], 0)
        self.assertAlmostEqual(la + np.log1p(row["to_usual"]["a"]) - beta * lb, last["mean"], places=4)
        self.assertAlmostEqual(la - beta * (lb + np.log1p(row["to_usual"]["b"])), last["mean"], places=4)
        self.assertAlmostEqual(row["usual_ratio"], row["ratio"] * (1 + row["to_usual"]["a"]), places=4)

    def test_the_replay_stops_where_live_tracking_starts(self):
        days = _weekdays(date(2026, 3, 2), 200)
        self.assertGreater(days[-1], cp.LIVE_SINCE)
        gold, silver = _cointegrated(200, shock_at=199)
        payload, ledger = cp.run(_series(days, gold, silver), [], datetime(2026, 12, 8, tzinfo=timezone.utc))
        row = next(p for p in payload["pairs"] if p["key"] == "gold-silver")

        self.assertEqual(payload["track_record"]["live_since"], cp.LIVE_SINCE.isoformat())
        self.assertTrue(all(e["entry_date"] < cp.LIVE_SINCE.isoformat() for e in row["replay_signals"]))
        self.assertEqual(row["replay"]["signals"], len(row["replay_signals"]))
        self.assertEqual([e["entry_date"] for e in ledger if e["pair"] == "gold-silver"], [days[-1].isoformat()])

    def test_a_negative_fit_falls_back_to_the_ratio(self):
        days = _weekdays(date(2025, 6, 2), 120)
        rng = np.random.default_rng(3)
        common = np.cumsum(rng.normal(0, 0.01, 120))
        gold, silver = np.exp(8 + common), np.exp(3 - common)
        row = cp.track_pair(GOLD_SILVER, _series(days, gold, silver))

        self.assertEqual((row["hedge"], row["beta"]), ("ratio", 1.0))

    def test_too_little_history_gives_a_note_instead_of_a_z_score(self):
        days = _weekdays(date(2026, 7, 6), 40)
        row = cp.track_pair(GOLD_SILVER, _series(days, np.full(40, 4000.0), np.full(40, 50.0)))

        self.assertNotIn("z", row)
        self.assertIn("40 sessions", row["note"])

    def test_history_uses_no_later_data(self):
        days = _weekdays(date(2025, 6, 2), 200)
        gold, silver = _cointegrated(200, seed=5)
        series = _series(days, gold, silver)
        d_all, la, lb = cp.aligned(GOLD_SILVER, series)
        full = cp.history(d_all, la, lb)
        cut = cp.history(d_all[:150], la[:150], lb[:150])

        self.assertEqual([round(h["z"], 10) for h in full[: len(cut)]], [round(h["z"], 10) for h in cut])


class LedgerTests(unittest.TestCase):
    def _payload(self, n, ledger, shock_at=None, seed=1):
        days = _weekdays(date(2025, 6, 2), n)
        gold, silver = _cointegrated(n, seed=seed, shock_at=shock_at)
        now = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        return cp.run(_series(days, gold, silver), ledger, now)

    def test_a_signal_opens_on_the_newest_session_and_closes_when_the_gap_closes(self):
        payload, ledger = self._payload(200, [], shock_at=199)
        gs = [e for e in ledger if e["pair"] == "gold-silver"]
        self.assertEqual(len(gs), 1)
        self.assertEqual(gs[0]["status"], "open")
        self.assertEqual(gs[0]["side"], -1)
        self.assertEqual(payload["track_record"]["by_pair"]["gold-silver"]["open"], 1)

        # The same history plus later sessions where the shock decays: the open signal resolves.
        _, ledger = self._payload(225, ledger, shock_at=199)
        gs = [e for e in ledger if e["pair"] == "gold-silver"]
        self.assertEqual(len(gs), 1)
        self.assertIn(gs[0]["status"], ("reverted", "timed_out"))
        self.assertLessEqual(gs[0]["sessions"], cp.MAX_HOLD)
        self.assertIn("spread_return", gs[0])

    def test_a_rescaled_history_leaves_a_signals_return_unchanged(self):
        # A roll ratio-adjusts every earlier MCX price, so a signal's move is measured on one basis.
        _, ledger = self._payload(200, [], shock_at=199)
        days = _weekdays(date(2025, 6, 2), 225)
        gold, silver = _cointegrated(225, shock_at=199)
        now = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        _, plain = cp.run(_series(days, gold, silver), [dict(e) for e in ledger], now)
        _, scaled = cp.run(_series(days, gold * 1.05, silver), [dict(e) for e in ledger], now)
        plain, scaled = (next(e for e in led if e["pair"] == "gold-silver") for led in (plain, scaled))

        self.assertIn(plain["status"], ("reverted", "timed_out"))
        self.assertAlmostEqual(scaled["spread_return"], plain["spread_return"], places=6)

    def test_rerunning_the_same_session_adds_nothing(self):
        _, ledger = self._payload(200, [], shock_at=199)
        _, again = self._payload(200, [dict(e) for e in ledger], shock_at=199)
        self.assertEqual(len(again), len(ledger))

    def test_a_quiet_pair_records_no_signal(self):
        payload, ledger = self._payload(200, [], seed=11)
        row = next(p for p in payload["pairs"] if p["key"] == "gold-silver")
        self.assertLess(abs(row["z"]), cp.ENTRY_Z)
        self.assertEqual([e for e in ledger if e["pair"] == "gold-silver"], [])

    def test_simulated_trades_pay_the_spread_move_net_of_costs(self):
        trades = cp.simulate(
            [date(2026, 1, d) for d in range(1, 4)], np.log([100.0, 100.0, 100.0]), np.log([1.0, 1.0, 1.0]),
            [{"i": 1, "date": date(2026, 1, 2), "beta": 1.0, "spread": 0.05, "z": 2.5},
             {"i": 2, "date": date(2026, 1, 3), "beta": 1.0, "spread": 0.0, "z": 0.1}],
        )
        self.assertEqual(trades[0]["status"], "reverted")
        # Entry spread 0.05, exit spread log(100) - log(1); the trade is short the spread.
        expected = -1 * (np.log(100) - 0.05) / 2 - 2 * cp.COST_PER_SIDE
        self.assertAlmostEqual(trades[0]["spread_return"], expected, places=6)


class TickScheduleTests(unittest.TestCase):
    def test_skips_once_the_newest_weekday_is_published(self):
        monday = datetime(2026, 11, 9, 3, 0, tzinfo=timezone.utc)
        previous = {"data_through": "2026-11-06", "generated_at": "2026-11-07T06:00:00+00:00"}
        self.assertTrue(tick.already_published(previous, monday))
        tuesday = monday + timedelta(days=1)
        self.assertFalse(tick.already_published(previous, tuesday))
        # No quota to protect: every run fetches until the new weekday is in.
        self.assertFalse(tick.already_published({**previous, "generated_at": "2026-11-10T01:00:00+00:00"}, tuesday))


if __name__ == "__main__":
    unittest.main()
