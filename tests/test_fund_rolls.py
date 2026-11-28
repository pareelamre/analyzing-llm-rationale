"""Futures near expiry: the next contract is traded and held positions roll (2026-11-28)."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_universe as fu  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_cycle_rolls", ROOT / "scripts" / "fund_cycle.py")
cycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cycle)

IST_OPEN = datetime(2026, 11, 30, 4, 1, tzinfo=timezone.utc)  # Monday 09:31 IST, expiry day


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def _leg(iid, key, contract, expiry, label):
    return {"id": iid, "label": label, "venue": "mcx", "asset_class": "commodity", "underlying": iid.split(":")[1].lower(),
            "currency": "INR", "multiplier": 1.0, "key": key, "expiry": expiry, "contract": contract}


class ContractChoiceTests(unittest.TestCase):
    def test_a_contract_in_its_last_week_gives_way_to_the_next(self):
        rows = [{"instrument_key": "NOV", "expiry": _ms(date(2026, 11, 30))},
                {"instrument_key": "DEC", "expiry": _ms(date(2026, 12, 31))}]
        self.assertEqual(fu.tradable(rows, date(2026, 11, 20))["instrument_key"], "NOV")
        self.assertEqual(fu.tradable(rows, date(2026, 11, 26))["instrument_key"], "DEC")
        self.assertEqual(fu.tradable(rows[:1], date(2026, 11, 29))["instrument_key"], "NOV")  # nothing later

    def test_the_universe_is_rebuilt_before_a_contract_expires_and_remembers_it(self):
        old = {"as_of": "2026-11-09", "usdinr": 92.0, "indices": [],
               "pairs": [{"key": "zinc-lead", "a": _leg("mcx:ZINCMINI", "Z-NOV", "ZINCMINI NOV", "2026-11-30", "Zinc"),
                          "b": _leg("mcx:LEADMINI", "L-NOV", "LEADMINI NOV", "2026-11-30", "Lead")}]}
        new = {"as_of": "2026-11-26", "usdinr": 92.0, "indices": [],
               "pairs": [{"key": "zinc-lead", "a": _leg("mcx:ZINCMINI", "Z-DEC", "ZINCMINI DEC", "2026-12-31", "Zinc"),
                          "b": _leg("mcx:LEADMINI", "L-DEC", "LEADMINI DEC", "2026-12-31", "Lead")}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fund_universe.json"
            path.write_text(json.dumps(old))
            with mock.patch.object(cycle, "UNIVERSE_PATH", path), \
                    mock.patch.object(cycle.fu, "build_universe", return_value=None), \
                    mock.patch.object(cycle.fu, "universe_payload", return_value=dict(new)):
                self.assertEqual(cycle.load_universe(date(2026, 11, 20), 92.0)["as_of"], "2026-11-09")  # 10 days left
                got = cycle.load_universe(date(2026, 11, 26), 92.0)
            self.assertEqual(got["as_of"], "2026-11-26")
            self.assertEqual(got["expiries"]["Z-NOV"], "2026-11-30")
            self.assertEqual(json.loads(path.read_text())["expiries"]["L-DEC"], "2026-12-31")


class RollTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        ledger.ensure_schema(self.conn)
        ledger.open_fund(self.conn, cycle.FUND_ID, 10_000.0)
        old_cu = _leg("mcx:COPPER", "CU-NOV", "COPPER NOV", "2026-11-30", "Copper")
        old_al = _leg("mcx:ALUMINI", "AL-NOV", "ALUMINI NOV", "2026-11-30", "Aluminium")
        for leg, side, price in ((old_cu, "LONG", 1300.0), (old_al, "SHORT", 330.0)):
            ledger.buy(self.conn, cycle.FUND_ID, leg, side, 20.0, price, fx=92.0, attribution={"pairs_model": 1.0},
                       pair_key="copper-aluminium", now="2026-11-16T08:01:17+00:00")
        self.universe = {"pairs": [{"key": "copper-aluminium",
                                    "a": _leg("mcx:COPPER", "CU-DEC", "COPPER DEC", "2026-12-31", "Copper"),
                                    "b": _leg("mcx:ALUMINI", "AL-DEC", "ALUMINI DEC", "2026-12-31", "Aluminium")}],
                         "expiries": {"CU-NOV": "2026-11-30", "AL-NOV": "2026-11-30"}}
        self.rates = {"USD": 1.0, "INR": 92.0}

    def _candles(self, prices):
        at = datetime(2026, 11, 30, 4, 0, tzinfo=timezone.utc)  # 09:30 IST, this session
        return lambda key, now, http_get=None: (at, prices[key], 100.0) if key in prices else None

    def test_both_legs_move_to_the_next_contract_at_the_same_value(self):
        prices = {"CU-NOV": 1250.0, "CU-DEC": 1260.0, "AL-NOV": 335.0, "AL-DEC": 338.0}
        before = ledger.nav(self.conn, cycle.FUND_ID, {"mcx:COPPER": {"last": 1250.0}, "mcx:ALUMINI": {"last": 335.0}},
                            self.rates)["nav_usd"]
        quotes = {}
        with mock.patch.object(cycle.fo, "latest_candle", side_effect=self._candles(prices)):
            done, waiting = cycle.roll_contracts(self.conn, self.universe, quotes, IST_OPEN, IST_OPEN.isoformat(),
                                                 self.rates, None)
        self.assertEqual((len(done), waiting), (2, []))
        held = {p["instrument_id"]: dict(p) for p in ledger.positions(self.conn, cycle.FUND_ID)}
        self.assertEqual({k: (p["key"], p["side"], p["pair_key"]) for k, p in held.items()},
                         {"mcx:COPPER": ("CU-DEC", "LONG", "copper-aluminium"),
                          "mcx:ALUMINI": ("AL-DEC", "SHORT", "copper-aluminium")})
        self.assertAlmostEqual(held["mcx:COPPER"]["quantity"], 20.0 * 1250.0 / 1260.0)
        self.assertEqual(held["mcx:COPPER"]["opened_at"], "2026-11-16T08:01:17+00:00")  # the holding clock runs on
        after = ledger.nav(self.conn, cycle.FUND_ID, quotes, self.rates)["nav_usd"]
        self.assertAlmostEqual(after, before, places=6)
        purposes = {r[0] for r in self.conn.execute("SELECT purpose FROM fund_trades WHERE ts = ?", (IST_OPEN.isoformat(),))}
        self.assertEqual(purposes, {"roll"})

    def test_a_pair_waits_together_when_one_new_contract_has_no_price_yet(self):
        prices = {"CU-NOV": 1250.0, "CU-DEC": 1260.0, "AL-NOV": 335.0}  # no December aluminium trade yet
        with mock.patch.object(cycle.fo, "latest_candle", side_effect=self._candles(prices)):
            done, waiting = cycle.roll_contracts(self.conn, self.universe, {}, IST_OPEN, IST_OPEN.isoformat(),
                                                 self.rates, None)
        self.assertEqual(done, [])
        self.assertEqual(len(waiting), 2)
        self.assertTrue(all("waiting to roll" in w["reason"] for w in waiting))
        self.assertEqual({dict(p)["key"] for p in ledger.positions(self.conn, cycle.FUND_ID)}, {"CU-NOV", "AL-NOV"})

    def test_an_expired_contract_rolls_at_its_last_close(self):
        stale = datetime(2026, 11, 30, 18, 0, tzinfo=timezone.utc)  # 23:30 IST on expiry day
        fresh = datetime(2026, 12, 1, 4, 0, tzinfo=timezone.utc)
        now = datetime(2026, 12, 1, 4, 1, tzinfo=timezone.utc)

        def candle(key, when, http_get=None):
            return {"CU-NOV": (stale, 1250.0, 1.0), "AL-NOV": (stale, 335.0, 1.0),
                    "CU-DEC": (fresh, 1260.0, 1.0), "AL-DEC": (fresh, 338.0, 1.0)}.get(key)
        with mock.patch.object(cycle.fo, "latest_candle", side_effect=candle):
            done, waiting = cycle.roll_contracts(self.conn, self.universe, {}, now, now.isoformat(), self.rates, None)
        self.assertEqual((len(done), waiting), (2, []))


if __name__ == "__main__":
    unittest.main()
