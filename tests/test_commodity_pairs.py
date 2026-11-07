from __future__ import annotations

import importlib.util
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


def _rows(days, gold, silver):
    rows = []
    for d, g, s in zip(days, gold, silver):
        rows.append({"date": d.isoformat(), "commodity": "gold", "international_price_usd": g,
                     "india_market_price_inr": g * 3.5, "india_landed_price_inr": g * 3.2})
        rows.append({"date": d.isoformat(), "commodity": "silver", "international_price_usd": s})
    return rows


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


class SeriesTests(unittest.TestCase):
    def test_weekends_and_stale_international_prices_are_dropped(self):
        rows = [
            {"date": "2026-11-06", "commodity": "zinc", "international_price_usd": 4050, "international_stale": True, "india_market_price_inr": 411.9},
            {"date": "2026-11-07", "commodity": "gold", "international_price_usd": 4074.9},
            {"date": "2026-11-06", "commodity": "gold", "international_price_usd": 4066.8, "india_market_price_inr": None},
        ]
        series = cp.price_series(rows)

        self.assertEqual(series[("gold", "international")], [(date(2026, 11, 6), 4066.8)])
        self.assertNotIn(("zinc", "international"), series)
        self.assertEqual(series[("zinc", "india")], [(date(2026, 11, 6), 411.9)])

    def test_parity_pairs_the_exchange_price_with_the_landed_cost(self):
        days = _weekdays(date(2026, 4, 6), 3)
        series = cp.price_series(_rows(days, [100, 101, 102], [1, 1, 1]))
        _, la, lb = cp.aligned(cp.PAIRS[-1], series)

        np.testing.assert_allclose(np.exp(la - lb), [3.5 / 3.2] * 3)


class TrackerTests(unittest.TestCase):
    def test_a_cointegrated_pair_reads_cointegrated_with_a_short_half_life(self):
        days = _weekdays(date(2025, 6, 2), 320)
        gold, silver = _cointegrated(320)
        row = cp.track_pair(GOLD_SILVER, cp.price_series(_rows(days, gold, silver)))

        self.assertEqual(row["hedge"], "regression")
        self.assertAlmostEqual(row["beta"], 0.9, delta=0.15)
        self.assertTrue(row["cointegrated"])
        self.assertLess(row["half_life"], 10)
        self.assertFalse(row["short_history"])
        self.assertEqual(len(row["z_recent"]), cp.Z_WINDOW)

    def test_a_shock_to_the_gap_makes_the_pair_stretched_with_the_rich_leg_named(self):
        days = _weekdays(date(2025, 6, 2), 200)
        gold, silver = _cointegrated(200, shock_at=199)
        row = cp.track_pair(GOLD_SILVER, cp.price_series(_rows(days, gold, silver)))

        self.assertGreater(row["z"], cp.ENTRY_Z)
        self.assertEqual((row["stretched"], row["rich"], row["cheap"]), (True, "Gold", "Silver"))

    def test_a_negative_fit_falls_back_to_the_ratio(self):
        days = _weekdays(date(2025, 6, 2), 120)
        rng = np.random.default_rng(3)
        common = np.cumsum(rng.normal(0, 0.01, 120))
        gold, silver = np.exp(8 + common), np.exp(3 - common)
        row = cp.track_pair(GOLD_SILVER, cp.price_series(_rows(days, gold, silver)))

        self.assertEqual((row["hedge"], row["beta"]), ("ratio", 1.0))

    def test_too_little_history_gives_a_note_instead_of_a_z_score(self):
        days = _weekdays(date(2026, 7, 6), 40)
        row = cp.track_pair(GOLD_SILVER, cp.price_series(_rows(days, np.full(40, 4000.0), np.full(40, 50.0))))

        self.assertNotIn("z", row)
        self.assertIn("40 sessions", row["note"])

    def test_history_uses_no_later_data(self):
        days = _weekdays(date(2025, 6, 2), 200)
        gold, silver = _cointegrated(200, seed=5)
        series = cp.price_series(_rows(days, gold, silver))
        d_all, la, lb = cp.aligned(GOLD_SILVER, series)
        full = cp.history(d_all, la, lb)
        cut = cp.history(d_all[:150], la[:150], lb[:150])

        self.assertEqual([round(h["z"], 10) for h in full[: len(cut)]], [round(h["z"], 10) for h in cut])


class LedgerTests(unittest.TestCase):
    def _payload(self, n, ledger, shock_at=None, seed=1):
        days = _weekdays(date(2025, 6, 2), n)
        gold, silver = _cointegrated(n, seed=seed, shock_at=shock_at)
        now = datetime.combine(days[-1] + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        return cp.run(_rows(days, gold, silver), ledger, now)

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
        self.assertTrue(tick.already_published({**previous, "generated_at": "2026-11-10T01:00:00+00:00"}, tuesday))


if __name__ == "__main__":
    unittest.main()
