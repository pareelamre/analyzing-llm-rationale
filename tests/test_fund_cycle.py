from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from analyzing_llm_rationale import benchmark_tools, fund_ledger  # noqa: E402

_spec = importlib.util.spec_from_file_location("fund_cycle", _ROOT / "scripts" / "fund_cycle.py")
cycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cycle)


class FundCycleTests(unittest.TestCase):
    def test_the_fund_opens_with_the_agents_cash_and_positions_at_its_entry_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent_path = Path(tmp) / "agent.sqlite"
            agent = sqlite3.connect(agent_path)
            benchmark_tools._ensure_account_schema(agent)
            agent.execute("INSERT INTO agent_accounts (agent_id, starting_cash, cash, updated_at) VALUES (?, ?, ?, ?)",
                          (cycle.SEED_AGENT_ID, 10_000.0, 6_774.06, "x"))
            agent.execute("INSERT INTO agent_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (cycle.SEED_AGENT_ID, "kalshi", "KXFED", "YES", 405.0, 356.40, 0.88, "x"))
            agent.execute("INSERT INTO agent_positions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (cycle.SEED_AGENT_ID, "polymarket", "iran-slug", "NO", 8598.31, 498.70, 0.058, "x"))
            agent.commit()
            fund = sqlite3.connect(":memory:")
            fund_ledger.ensure_schema(fund)
            with (
                mock.patch.object(cycle.market_data, "fetch_kalshi", return_value={"question": "Fed cuts in December?"}),
                mock.patch.object(cycle.market_data, "fetch_polymarket",
                                  side_effect=cycle.market_data.MarketDataError("gone")),
            ):
                seeded = cycle.seed_from_agent(fund, agent_path, "2026-11-09T00:00:00+00:00")
            self.assertEqual(seeded["positions"], 2)
            snap = fund_ledger.nav(fund, cycle.FUND_ID, {}, {})
            self.assertAlmostEqual(snap["cash_usd"], 6_774.06)
            self.assertAlmostEqual(snap["nav_usd"], 6_774.06 + 356.40 + 498.70)
            self.assertEqual(snap["starting_value"], 10_000.0)
            labels = {p["instrument_id"]: p["label"] for p in snap["positions"]}
            # A market whose question can't be fetched keeps its ticker as the label.
            self.assertEqual(labels, {"kalshi:KXFED": "Fed cuts in December?", "polymarket:iran-slug": "iran-slug"})
            actions = {r[0] for r in fund.execute("SELECT action FROM fund_trades")}
            self.assertEqual(actions, {"transfer"})

    def test_recent_rationale_checks_become_a_shadow_analyst_with_its_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = Path(tmp) / "checks.json"
            checks.write_text(json.dumps({"checks": [
                {"instrument_id": "polymarket:iran", "question": "Iran?", "forecast": 0.98, "rationale_yes": 0.04,
                 "verdict": "contradicts", "forecasted_at": "2026-11-09T17:25:00+00:00"},
                {"instrument_id": "kalshi:KXFED", "question": "Fed?", "forecast": 0.3, "rationale_yes": 0.2,
                 "verdict": "agrees", "forecasted_at": "2026-11-09T17:26:00+00:00"},
                {"instrument_id": "kalshi:OLD", "question": "Old?", "forecast": 0.9, "rationale_yes": 0.1,
                 "verdict": "contradicts", "forecasted_at": "2026-11-01T00:00:00+00:00"},
            ]}))
            fund = sqlite3.connect(":memory:")
            fund_ledger.ensure_schema(fund)
            with mock.patch.object(cycle, "CHECKS_PATH", checks), \
                    mock.patch.object(cycle, "BLEND_PATH", Path(tmp) / "absent.json"):
                shadow = cycle.record_shadow(fund, [], datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc),
                                             "2026-11-10T00:00:00+00:00")
        checker = shadow["rationale_check"]
        self.assertEqual(checker["markets"], 2)
        self.assertEqual([f["label"] for f in checker["flags"]], ["Iran?"])
        self.assertEqual(fund.execute("SELECT COUNT(*) FROM analyst_predictions").fetchone()[0], 2)

    def test_polymarket_no_is_the_mirror_of_yes(self):
        quote = cycle.event_quote({"yes_bid": 0.04, "yes_ask": 0.06, "no_bid": None, "no_ask": None})
        self.assertAlmostEqual(quote["no_bid"], 0.94)
        self.assertAlmostEqual(quote["no_ask"], 0.96)

    def test_trading_waits_for_its_hours_and_its_cooldown(self):
        fund = sqlite3.connect(":memory:")
        fund_ledger.ensure_schema(fund)
        with mock.patch.dict(os.environ, {"FUND_ANY_HOUR": ""}):
            # 20:00 UTC is 01:30 IST, inside the window.
            self.assertTrue(cycle.trading_window(fund, datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc))[0])
            # 22:00 UTC is 03:30 IST, outside it.
            self.assertFalse(cycle.trading_window(fund, datetime(2026, 11, 9, 22, 0, tzinfo=timezone.utc))[0])
            fund_ledger.open_fund(fund, cycle.FUND_ID, 100.0)
            fund_ledger.buy(fund, cycle.FUND_ID, {"id": "kalshi:X"}, "YES", 1, 0.5, cycle_id="fund-20261109T2000",
                            now="2026-11-09T20:00:00+00:00")
            ok, why = cycle.trading_window(fund, datetime(2026, 11, 9, 20, 45, tzinfo=timezone.utc))
            self.assertFalse(ok)
            self.assertIn("under 10 hours ago", why)

    def test_an_expired_structure_is_settled_at_the_index_close_without_blocking_trading(self):
        from analyzing_llm_rationale import fund_options as fo

        condor = fo.Structure("condor", "RANGE", [
            fo.Leg("p1", "PE", 24_300, -1), fo.Leg("p0", "PE", 24_000, 1),
            fo.Leg("c1", "CE", 25_700, -1), fo.Leg("c0", "CE", 26_000, 1)], const=300.0)
        spec = fo.instrument("NIFTY", condor, datetime(2026, 11, 17).date())
        fund = sqlite3.connect(":memory:")
        fund_ledger.ensure_schema(fund)
        fund_ledger.open_fund(fund, cycle.FUND_ID, 10_000.0)
        fund_ledger.buy(fund, cycle.FUND_ID, spec, "RANGE", 100, 160.0, fx=88.0, cycle_id="fund-20261110T0800",
                        now="2026-11-10T08:00:00+00:00")
        specs = {spec["id"]: fund_ledger.instrument_spec(fund, cycle.FUND_ID, spec["id"])}
        before = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)  # 14:30 IST on expiry day: not yet
        with mock.patch.object(fo, "settlement_level", return_value=24_150.0) as level:
            self.assertEqual(cycle.settle_expired(fund, dict(specs), before, before.isoformat(), {"INR": 88.0}), [])
            level.assert_not_called()
            after = datetime(2026, 11, 17, 10, 30, tzinfo=timezone.utc)  # 16:00 IST
            [done] = cycle.settle_expired(fund, specs, after, after.isoformat(), {"INR": 88.0})
        self.assertEqual(done["value"], 150.0)  # half the put spread lost
        self.assertEqual(fund_ledger.positions(fund, cycle.FUND_ID), [])
        trade = fund.execute("SELECT action, price, realized_pnl_usd, cycle_id FROM fund_trades"
                             " ORDER BY ts DESC LIMIT 1").fetchone()
        self.assertEqual((trade[0], trade[1], trade[3]), ("expire", 150.0, "expiry-2026-11-17"))
        self.assertAlmostEqual(trade[2], (150.0 - 160.0) * 100 / 88.0)
        # Settling is not a trading cycle, so it never holds back the next one.
        self.assertEqual(cycle._last_cycle(fund), datetime.fromisoformat("2026-11-10T08:00:00+00:00"))

    def test_settled_kalshi_positions_pay_a_dollar_to_the_winning_side(self):
        fund = sqlite3.connect(":memory:")
        fund_ledger.ensure_schema(fund)
        fund_ledger.open_fund(fund, cycle.FUND_ID, 1_000.0)
        for ticker, side in (("KXHIGHNY-26NOV10-T61", "NO"), ("KXHIGHNY-26NOV10-B61.5", "YES"),
                             ("KXHIGHNY-26NOV11-T61", "NO")):
            fund_ledger.buy(fund, cycle.FUND_ID, {"id": f"kalshi:{ticker}", "venue": "kalshi", "label": ticker},
                            side, 100, 0.40)
        results = {"KXHIGHNY-26NOV10-T61": "no", "KXHIGHNY-26NOV10-B61.5": "no", "KXHIGHNY-26NOV11-T61": None}
        now = datetime(2026, 11, 11, 8, 0, tzinfo=timezone.utc)
        with mock.patch.object(cycle.market_data, "fetch_kalshi_result", side_effect=results.get):
            done = cycle.settle_events(fund, now, now.isoformat())
        self.assertEqual({d["instrument"]: d["value"] for d in done},
                         {"kalshi:KXHIGHNY-26NOV10-T61": 1.0, "kalshi:KXHIGHNY-26NOV10-B61.5": 0.0})
        self.assertEqual([p["instrument_id"] for p in fund_ledger.positions(fund, cycle.FUND_ID)],
                         ["kalshi:KXHIGHNY-26NOV11-T61"])  # not settled yet
        realized = fund.execute("SELECT realized_pnl FROM fund_account").fetchone()[0]
        self.assertAlmostEqual(realized, 100 * (1 - 0.40) - 100 * 0.40)
        self.assertIsNone(cycle._last_cycle(fund))  # settling is not a trading cycle

    def test_the_weather_analyst_trades_only_on_a_go_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            with mock.patch.object(cycle, "WEATHER_REPORT_PATH", path):
                self.assertEqual(cycle.weather_mode()[0], "SHADOW")  # no report yet
                path.write_text(json.dumps({"verdict": "SHADOW"}))
                self.assertEqual(cycle.weather_mode()[0], "SHADOW")
                path.write_text(json.dumps({"verdict": "GO"}))
                self.assertEqual(cycle.weather_mode()[0], "GO")


if __name__ == "__main__":
    unittest.main()
