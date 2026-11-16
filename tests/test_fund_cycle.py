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

    def test_the_guard_records_recent_checks_and_every_forecast_it_dropped(self):
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
            board = [{"platform": "Polymarket", "ident": "iran", "question": "Iran?", "model_probability": 0.98,
                      "market_bid": 0.04, "market_ask": 0.06, "forecasted_at": "2026-11-09T17:25:00+00:00"}]
            fund = sqlite3.connect(":memory:")
            fund_ledger.ensure_schema(fund)
            with mock.patch.object(cycle, "CHECKS_PATH", checks):
                _kept, guard = cycle.guard_board(board)
                guard = cycle.record_guard(fund, guard, datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc),
                                           "2026-11-10T00:00:00+00:00")
        self.assertEqual((guard["mode"], guard["markets"], guard["days_recorded"]), ("live", 2, 1))
        self.assertEqual([f["label"] for f in guard["flags"]], ["Iran?"])
        self.assertEqual(guard["dropped"], [{"label": "Iran?", "forecast": 0.98, "rationale_yes": 0.04,
                                             "forecasted_at": "2026-11-09T17:25:00+00:00"}])
        rows = [tuple(r) for r in fund.execute("SELECT analyst, mode, instrument_id, probability, market_probability"
                                               " FROM analyst_predictions ORDER BY analyst, instrument_id")]
        # The dropped forecast keeps its raw number and the market's price then, so it can be graded.
        self.assertEqual(rows, [("rationale_check", "live", "kalshi:KXFED", 0.2, None),
                                ("rationale_check", "live", "polymarket:iran", 0.04, None),
                                ("rationale_dropped", "live", "polymarket:iran", 0.98, 0.05)])

    def test_mcx_is_priced_at_this_sessions_bar_and_perps_say_if_they_are_open(self):
        # 2026-11-16: copper was bought at Friday's daily close while it traded 1.6% lower.
        now = datetime(2026, 11, 16, 8, 1, tzinfo=timezone.utc)
        universe = {"pairs": [{"a": {"id": "mcx:COPPER", "key": "MCX_FO|578634"},
                               "b": {"id": "kalshi_perp:KXGOLDPERP", "key": "KXGOLDPERP"}}]}
        bar = (datetime(2026, 11, 16, 13, 30, tzinfo=cycle.IST), 1288.0, 5000.0)
        perps = [{"ticker": "KXGOLDPERP", "bid": 0.1, "ask": 0.2, "schedule": {"is_open": False}}]
        with mock.patch.object(cycle.fo, "latest_candle", return_value=bar) as latest, \
                mock.patch.object(cycle.fu, "kalshi_margin_markets", return_value=perps), \
                mock.patch.object(cycle.fu, "kalshi_quote", return_value={"bid": 0.1, "ask": 0.2}):
            quotes = cycle.fetch_quotes({"mcx:COPPER", "kalshi_perp:KXGOLDPERP"}, universe, now)
        self.assertEqual(latest.call_args.args[:2], ("MCX_FO|578634", now))
        self.assertEqual(quotes["mcx:COPPER"], {"bid": 1288.0, "ask": 1288.0, "last": 1288.0,
                                                "at": "2026-11-16T13:30:00+05:30", "source": "last trade"})
        self.assertIs(quotes["kalshi_perp:KXGOLDPERP"]["open"], False)

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

    def test_prediction_markets_trade_on_the_blend_not_the_raw_model(self):
        # Owner, 2026-11-15: the raw on-device model trails the market, so its odds no longer trade.
        params = {"intercept": 0.173417, "coef": [0.693864, 0.402811, -0.291052]}
        row = {"platform": "Polymarket", "ident": "x", "question": "Will x?", "model_probability": 0.9,
               "market_bid": 0.49, "market_ask": 0.51, "forecasted_at": "2026-11-09T17:25:00+00:00"}
        now = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)
        raw, raw_skipped = cycle.orch.event_signals([row], now)
        self.assertEqual(raw, [])  # 40 points from the market: the raw model would be skipped
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blend.json"
            path.write_text(json.dumps(params))
            with mock.patch.object(cycle, "BLEND_PATH", path), \
                    mock.patch.object(cycle, "CHECKS_PATH", Path(tmp) / "no_checks.json"):
                signals, skipped, forecasts, guard = cycle.prediction_market_signals([row], now)
            with mock.patch.object(cycle, "BLEND_PATH", Path(tmp) / "absent.json"), \
                    mock.patch.object(cycle, "CHECKS_PATH", Path(tmp) / "no_checks.json"):
                none, why, no_forecasts, _ = cycle.prediction_market_signals([row], now)
        self.assertEqual([(s.analyst, s.side) for s in signals], [("blend_model", "YES")])
        [forecast] = forecasts.values()
        self.assertAlmostEqual(forecast["model_probability"], 0.603, places=2)
        self.assertEqual(forecast["raw_model_probability"], 0.9)
        self.assertEqual((none, no_forecasts, why[0]["reason"]), ([], {}, "no blend parameters"))
        # With no checks yet, the forecast is used as it is and counted as unchecked.
        self.assertEqual((guard["board"], guard["checked"], guard["unchecked"], guard["dropped"]), (1, 0, 1, []))

    def _guarded(self, row: dict, checks: list, now: datetime) -> tuple:
        params = {"intercept": 0.173417, "coef": [0.693864, 0.402811, -0.291052]}
        with tempfile.TemporaryDirectory() as tmp:
            blend, checks_path = Path(tmp) / "blend.json", Path(tmp) / "checks.json"
            blend.write_text(json.dumps(params))
            checks_path.write_text(json.dumps({"checks": checks}))
            with mock.patch.object(cycle, "BLEND_PATH", blend), mock.patch.object(cycle, "CHECKS_PATH", checks_path):
                return cycle.prediction_market_signals([row], now)

    def test_a_forecast_that_contradicts_its_own_rationale_never_reaches_the_blend(self):
        # Owner, 2026-11-16: the checker is a live guard. Unguarded, this row is a blend YES signal.
        row = {"platform": "Polymarket", "ident": "x", "question": "Will x?", "model_probability": 0.9,
               "market_bid": 0.49, "market_ask": 0.51, "forecasted_at": "2026-11-09T17:25:00.123456+00:00"}
        now = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)
        check = {"instrument_id": "polymarket:x", "question": "Will x?", "forecast": 0.9, "rationale_yes": 0.05,
                 "verdict": "contradicts", "forecasted_at": "2026-11-09T17:25:00.123456Z"}
        signals, skipped, forecasts, guard = self._guarded(row, [check], now)
        self.assertEqual(signals, [])
        # No forecast for the market either, so a held position is not sold on it.
        self.assertEqual(forecasts, {})
        self.assertEqual(skipped, [{"instrument": "polymarket:x", "label": "Will x?", "reason": cycle.GUARD_REASON}])
        self.assertEqual((guard["checked"], guard["unchecked"]), (1, 0))
        self.assertEqual([(d["ident"], d["model_probability"], d["rationale_yes"]) for d in guard["dropped"]],
                         [("x", 0.9, 0.05)])

    def test_the_guard_drops_only_the_forecast_it_read(self):
        row = {"platform": "Polymarket", "ident": "x", "question": "Will x?", "model_probability": 0.9,
               "market_bid": 0.49, "market_ask": 0.51, "forecasted_at": "2026-11-09T17:25:00+00:00"}
        now = datetime(2026, 11, 9, 20, 0, tzinfo=timezone.utc)
        older = {"instrument_id": "polymarket:x", "forecast": 0.9, "rationale_yes": 0.05, "verdict": "contradicts",
                 "forecasted_at": "2026-11-08T17:25:00+00:00"}
        other = {**older, "instrument_id": "polymarket:y", "forecasted_at": row["forecasted_at"]}
        agrees = {**older, "rationale_yes": 0.95, "verdict": "agrees", "forecasted_at": row["forecasted_at"]}
        unclear = {**agrees, "rationale_yes": 0.5, "verdict": "unclear"}
        for checks, checked in (([older, other], 0), ([agrees], 1), ([unclear], 1)):
            with self.subTest(verdicts=[c["verdict"] for c in checks]):
                signals, _skipped, _forecasts, guard = self._guarded(row, checks, now)
                self.assertEqual([(s.analyst, s.side) for s in signals], [("blend_model", "YES")])
                self.assertEqual((guard["checked"], guard["unchecked"], guard["dropped"]), (checked, 1 - checked, []))

    def test_the_scorecard_scale_is_read_per_analyst(self):
        card = {"strategies": [{"analyst": "put_premium", "scale": 2.5}, {"analyst": "blend_model", "scale": 0.5}]}
        self.assertEqual(cycle._scale(card, "put_premium"), 2.5)
        self.assertEqual(cycle._scale(card, "someone_else"), 1.0)
        self.assertEqual(cycle._scale({}, "put_premium"), 1.0)

    def test_the_retired_analysts_no_longer_run(self):
        self.assertEqual([r["analyst"] for r in cycle.RETIRED],
                         ["event_model", "weather_model", "decided_ranges", "options_model", "vix_fade"])
        for gone in ("fpl", "fvf", "weather_mode", "options_mode"):
            self.assertFalse(hasattr(cycle, gone))

    def test_the_put_sleeve_trades_only_while_its_out_of_sample_check_says_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            with mock.patch.object(cycle, "PUT_SLEEVE_REPORT_PATH", path):
                self.assertEqual(cycle.put_sleeve_mode(), "OFF")
                path.write_text(json.dumps({"put_spread": {"verdict": "NO-GO"}}))
                self.assertEqual(cycle.put_sleeve_mode(), "OFF")
                path.write_text(json.dumps({"put_spread": {"verdict": "GO"}}))
                self.assertEqual(cycle.put_sleeve_mode(), "GO")

    def test_bank_nifty_joins_the_sleeve_only_while_both_tests_say_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, bank = Path(tmp) / "ndx.json", Path(tmp) / "bank.json"
            with mock.patch.object(cycle, "PUT_SLEEVE_REPORT_PATH", base), \
                    mock.patch.object(cycle, "BANKNIFTY_SLEEVE_REPORT_PATH", bank):
                bank.write_text(json.dumps({"put_spread": {"verdict": "GO"}}))
                self.assertEqual(cycle.put_sleeve_modes(), {"NIFTY": "OFF", "BANKNIFTY": "OFF"})
                base.write_text(json.dumps({"put_spread": {"verdict": "GO"}}))
                self.assertEqual(cycle.put_sleeve_modes(), {"NIFTY": "GO", "BANKNIFTY": "GO"})
                bank.write_text(json.dumps({"put_spread": {"verdict": "NO-GO"}}))
                self.assertEqual(cycle.put_sleeve_modes(), {"NIFTY": "GO", "BANKNIFTY": "OFF"})
        self.assertEqual(cycle.put_sleeve_modes()["BANKNIFTY"], "GO")  # the repo's own reports

    def test_a_held_spread_blocks_only_its_own_index(self):
        self.assertEqual(cycle.sleeve_symbol("nse_fo:BANKNIFTY:put_spread:2026-12-29:57000/55000"), "BANKNIFTY")
        self.assertEqual(cycle.sleeve_symbol("nse_fo:NIFTY:put_spread:2026-12-08:24500/24000"), "NIFTY")
        self.assertEqual(cycle.sleeve_symbol("odd"), "")


if __name__ == "__main__":
    unittest.main()
