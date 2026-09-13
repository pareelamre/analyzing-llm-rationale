"""Close a losing position before its market resolves, when the record says it pays.

Positions held to settlement won 4 of 26 and lost $3,984 -- 86% of the fleet's
realized losses. A backtest on the fleet's own price paths (99 positions, exits
at the executable bid, fees charged, no lookahead) tested three families of
exit rule, and an adversarial reviewer re-ran each:

    fixed price stop-loss          at best a wash against what agents did
    trailing stop / combinations   worse by $578-$1,053 (whipsaw on news markets)
    down > 30% within 24h          +$302 on the current accounts, 1 winner cut

Only the last helped, so only the last is applied. Its size is fragile -- 43%
of the gain is one weather market -- but its direction held in both eras.

What these tests hold:
  - the rule selects exactly the positions it describes, on the side held,
    and nothing that has already closed, is winning, or has no bid;
  - an exit goes through the real close path and is recorded as a rule exit
    in the audit, where an agent cannot forge that label;
  - an order that fills nothing is never reported as a close;
  - the agent is told what the rule did, and the kill switch turns it off.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agent_trading_tick  # noqa: E402

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402
from tests.test_agent_trading_tick import _quote  # noqa: E402

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)


def iso(hours_from_now):
    return (NOW + timedelta(hours=hours_from_now)).isoformat().replace("+00:00", "Z")


def held(side="yes", quantity=100.0, avg=0.40, ticker="KXHELD", platform="kalshi"):
    return {
        "platform": platform, "ticker": ticker, "side": side,
        "quantity": quantity, "avg_entry_price": avg, "cost_basis": quantity * avg,
    }


def pick(position, quote):
    return agent_trading_tick._pre_expiry_exit_candidates([position], [quote], now=NOW)


class CandidateSelectionTests(unittest.TestCase):
    def test_a_position_down_forty_percent_with_hours_left_is_selected(self):
        chosen = pick(held(), _quote("KXHELD", bid=0.24, ask=0.26, close=iso(10)))
        self.assertEqual(len(chosen), 1)
        self.assertAlmostEqual(chosen[0]["change"], -0.40)
        self.assertAlmostEqual(chosen[0]["hours_left"], 10.0)

    def test_the_threshold_is_strict_as_backtested(self):
        """The backtest fired on price < 0.7 x avg entry: more than 30% down.

        Exactly 30% down (bid 0.28 on a 0.40 entry) stays with the agent; one
        cent further (0.27) is closed. Written as a price test, not
        quantity x bid against cost basis, because 100 x 0.28 is
        28.000000000000004 and that float noise decides a position sitting
        on the line.
        """
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.28, ask=0.30, close=iso(10))), [])
        self.assertEqual(len(pick(held(), _quote("KXHELD", bid=0.27, ask=0.29, close=iso(10)))), 1)

    def test_a_bid_exactly_on_the_line_stays_with_the_agent(self):
        """0.7 x 0.50 is exactly 0.35 in floating point, so this is the case
        that separates `<` from `<=`; the 0.40 entry above cannot, because
        0.7 x 0.40 is 0.27999999999999997 and 0.28 fails either form."""
        self.assertEqual(0.7 * 0.50, 0.35)
        self.assertEqual(pick(held(avg=0.50), _quote("KXHELD", bid=0.35, ask=0.37, close=iso(10))), [])
        self.assertEqual(len(pick(held(avg=0.50), _quote("KXHELD", bid=0.34, ask=0.36, close=iso(10)))), 1)

    def test_the_twenty_four_hour_edge_is_inclusive(self):
        """The backtest counted a point exactly H hours before close as inside."""
        self.assertEqual(len(pick(held(), _quote("KXHELD", bid=0.10, ask=0.12, close=iso(24)))), 1)

    def test_a_smaller_loss_is_left_to_the_agent(self):
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.30, ask=0.32, close=iso(10))), [])

    def test_a_market_more_than_a_day_out_is_left_alone(self):
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.10, ask=0.12, close=iso(30))), [])

    def test_a_market_that_has_already_closed_cannot_be_exited(self):
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.10, ask=0.12, close=iso(-2))), [])

    def test_a_winning_position_is_never_selected(self):
        self.assertEqual(pick(held(avg=0.20), _quote("KXHELD", bid=0.60, ask=0.62, close=iso(5))), [])

    def test_a_no_holding_is_judged_on_the_no_bid(self):
        # NO bid = 1 - YES ask = 0.20 -> 100 NO at avg 0.50 is down 60%.
        self.assertEqual(len(pick(held(side="no", avg=0.50), _quote("KXHELD", bid=0.78, ask=0.80, close=iso(6)))), 1)
        # Read off the YES bid (0.78) the same holding would look like a gain.
        self.assertEqual(pick(held(side="no", avg=0.50), _quote("KXHELD", bid=0.40, ask=0.42, close=iso(6))), [])

    def test_no_executable_bid_means_no_exit_attempt(self):
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.0, ask=0.02, close=iso(6))), [])

    def test_no_quote_means_no_exit_attempt(self):
        self.assertEqual(agent_trading_tick._pre_expiry_exit_candidates([held()], [], now=NOW), [])

    def test_an_unknown_close_time_means_no_exit_attempt(self):
        self.assertEqual(pick(held(), _quote("KXHELD", bid=0.10, ask=0.12, close="whenever")), [])


class _AccountCase(unittest.TestCase):
    AGENT = "model-exit"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(self.tmp.name) / "accounts.sqlite"),
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(self.tmp.name) / "ledger.jsonl"),
            "FORESEA_AGENT_NOTES_PATH": str(Path(self.tmp.name) / "notes.json"),
            "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
        }
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("FORESEA_AGENT_PRE_EXPIRY_EXIT", None)

    def open_yes(self, ticker="KXHELD", quantity=10):
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote(ticker, bid=0.40, ask=0.42)):
            result = benchmark_tools.place_trade(
                {"ticker": ticker, "side": "yes", "price": 0.42, "quantity": quantity},
                benchmark_tools.ToolContext(agent_id=self.AGENT),
            )
        self.assertTrue(result["ok"], result)

    def held_quantity(self, ticker="KXHELD"):
        with benchmark_tools._account_transaction() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(quantity), 0) AS q FROM agent_positions WHERE agent_id = ? AND ticker = ?",
                (self.AGENT, ticker),
            ).fetchone()
        return float(row["q"])

    def audits(self, ticker="KXHELD"):
        with benchmark_tools._account_transaction() as conn:
            rows = conn.execute(
                "SELECT metadata_json FROM agent_actions WHERE agent_id = ? AND ticker = ? ORDER BY ts",
                (self.AGENT, ticker),
            ).fetchall()
        return [json.loads(r["metadata_json"] or "{}").get("audit") or {} for r in rows]

    def run_exits(self, live_quote):
        held_quote = dict(live_quote)
        with mock.patch.object(market_data, "fetch_kalshi", return_value=live_quote):
            return agent_trading_tick._run_pre_expiry_exits(self.AGENT, [held_quote], now=NOW)


class RunExitsTests(_AccountCase):
    def test_a_qualifying_position_is_closed_and_marked_as_a_rule_exit(self):
        self.open_yes()
        outcomes = self.run_exits(_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5)))
        self.assertEqual([o["status"] for o in outcomes], ["closed"])
        self.assertAlmostEqual(self.held_quantity(), 0.0, places=6)
        tags = [a.get("initiated_by") for a in self.audits()]
        self.assertEqual(tags, [None, agent_trading_tick.PRE_EXPIRY_EXIT_RULE])

    def test_a_position_that_does_not_qualify_is_not_touched(self):
        self.open_yes()
        self.assertEqual(self.run_exits(_quote("KXHELD", bid=0.39, ask=0.41, close=iso(5))), [])
        self.assertAlmostEqual(self.held_quantity(), 10.0, places=6)

    def test_the_kill_switch_turns_the_rule_off(self):
        self.open_yes()
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_PRE_EXPIRY_EXIT": "off"}):
            self.assertEqual(self.run_exits(_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))), [])
        self.assertAlmostEqual(self.held_quantity(), 10.0, places=6)

    def test_an_exit_that_fills_nothing_is_not_reported_as_closed(self):
        self.open_yes()
        held_quote = _quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))
        with mock.patch.object(market_data, "fetch_kalshi", side_effect=market_data.MarketDataError("venue down")):
            outcomes = agent_trading_tick._run_pre_expiry_exits(self.AGENT, [held_quote], now=NOW)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "failed")
        self.assertEqual(outcomes[0]["detail"], "shadow_quote_unavailable")
        self.assertAlmostEqual(self.held_quantity(), 10.0, places=6)

    def _classify(self, trade_result):
        """Run the exit with place_trade returning a fixed result."""
        self.open_yes()
        held_quote = _quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))
        with mock.patch.object(benchmark_tools, "place_trade", return_value=trade_result):
            return agent_trading_tick._run_pre_expiry_exits(self.AGENT, [held_quote], now=NOW)[0]

    def test_an_order_accepted_with_nothing_filled_is_unfilled_not_closed(self):
        """place_trade returns ok for a zero fill; ok alone must not read as a close."""
        outcome = self._classify({
            "ok": True, "execution": {"filled_quantity": 0.0, "fill_status": "shadow_unfilled_no_depth"},
        })
        self.assertEqual(outcome["status"], "unfilled")
        self.assertEqual(outcome["detail"], "shadow_unfilled_no_depth")

    def test_an_order_that_fills_part_of_the_position_is_partial(self):
        outcome = self._classify({
            "ok": True, "execution": {"filled_quantity": 4.0, "fill_status": "shadow_filled_partial"},
        })
        self.assertEqual(outcome["status"], "partial")
        self.assertEqual(outcome["filled_quantity"], 4.0)

    def test_an_order_that_fills_the_whole_position_is_closed(self):
        outcome = self._classify({
            "ok": True, "execution": {"filled_quantity": 10.0, "fill_status": "shadow_filled_full"},
        })
        self.assertEqual(outcome["status"], "closed")

    def test_the_exit_order_is_a_reduce_only_close_of_the_opposite_side(self):
        """No quantity or price, so place_trade snaps to the exact held size."""
        self.open_yes()
        held_quote = _quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))
        with mock.patch.object(benchmark_tools, "place_trade", return_value={"ok": False}) as trade:
            agent_trading_tick._run_pre_expiry_exits(self.AGENT, [held_quote], now=NOW)
        order, ctx = trade.call_args[0]
        self.assertEqual(order, {"platform": "kalshi", "ticker": "KXHELD", "side": "no", "sizing_mode": "close"})
        self.assertEqual(ctx.initiated_by, agent_trading_tick.PRE_EXPIRY_EXIT_RULE)
        self.assertTrue(ctx.require_kelly_sizing)

    def test_an_agent_cannot_label_its_own_trade_a_rule_exit(self):
        """The tag comes from ToolContext, which a tool call cannot set."""
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXSPOOF", bid=0.40, ask=0.42)):
            benchmark_tools.place_trade(
                {"ticker": "KXSPOOF", "side": "yes", "price": 0.42, "quantity": 5,
                 "initiated_by": agent_trading_tick.PRE_EXPIRY_EXIT_RULE},
                benchmark_tools.ToolContext(agent_id=self.AGENT),
            )
        self.assertEqual([a.get("initiated_by") for a in self.audits("KXSPOOF")], [None])


class ExitNotesTests(unittest.TestCase):
    def test_the_agent_is_told_what_was_closed_and_what_was_not(self):
        base = {"ticker": "KXA", "side": "yes", "quantity": 10.0, "bid": 0.10, "change": -0.75, "hours_left": 5.0}
        notes = agent_trading_tick._fmt_pre_expiry_exits([
            {**base, "status": "closed", "filled_quantity": 10.0, "detail": ""},
            {**base, "ticker": "KXB", "status": "partial", "filled_quantity": 4.0, "detail": "shadow_filled_partial"},
            {**base, "ticker": "KXC", "status": "unfilled", "filled_quantity": 0.0, "detail": "shadow_quote_unavailable"},
        ])
        self.assertIn("Automatic pre-expiry exits this cycle", notes)
        self.assertIn("KXA YES", notes)
        self.assertIn("-> closed.", notes)
        self.assertIn("only 4.0 filled; the rest is still open", notes)
        self.assertIn("NOT closed (unfilled: shadow_quote_unavailable); still open", notes)

    def test_nothing_to_report_adds_nothing(self):
        self.assertIsNone(agent_trading_tick._fmt_pre_expiry_exits([]))


class RunCycleTests(_AccountCase):
    AGENT = "model-exit-cycle"

    def test_the_cycle_exits_before_the_agent_decides_and_says_so(self):
        self.open_yes()
        near_close = _quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))
        near_close["close_time"] = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
        with (
            mock.patch.dict(os.environ, {"FORESEA_AGENT_CYCLE_ID": "exit-cycle"}),
            mock.patch.object(agent_trading_tick, "_init_local_agent"),
            mock.patch.object(market_data, "fetch_kalshi", return_value=near_close),
            mock.patch.object(market_data, "resolve_kalshi", return_value=None),
            mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXNEW")]),
            mock.patch.object(market_data, "list_polymarket", return_value=[]),
            mock.patch.object(
                agent_trading_tick, "_call_agent_analyze",
                return_value=SimpleNamespace(thesis="Held.", tool_transcript=[]),
            ) as call_mock,
        ):
            agent_trading_tick.run_cycle(self.AGENT)
        question = call_mock.call_args[0][0]
        self.assertIn("Automatic pre-expiry exits this cycle", question)
        self.assertIn("KXHELD YES", question)
        self.assertIn("-> closed.", question)
        self.assertIn("Open positions: none.", question)
        self.assertNotIn("Markets you currently hold", question)
        self.assertAlmostEqual(self.held_quantity(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
