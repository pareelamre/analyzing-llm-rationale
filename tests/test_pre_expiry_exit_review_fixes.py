"""What an adversarial review of the pre-expiry exit rule found, held in place.

Eight defects were confirmed before the rule shipped; every verifier had been
told to default to "not real". Seven are fixed here. The eighth -- the
Datastore account backend dropping the whole audit block -- predates this
rule, affects every trade on the Cloud Run tool loop rather than this rule
(the scheduled tick always runs on SQLite), and is tracked separately; the
rule now refuses to run without a SQLite store instead.

  1. Polymarket was judged on its order-book bestBid, but the backtest only had
     Polymarket last-trade marks. The rule is Kalshi-only until Polymarket is
     tested on the price the rule would use.
  2. Kalshi NO holdings were judged on the venue's no_bid, but every backtested
     NO point was 1 - yes_ask. The rule now uses the backtested series.
  3. A rule exit used up the agent's per-cycle trade count, spend, daily risk
     and duplicate cooldown -- blocking the agent's own trades in the very
     cycle the exit freed capital for.
  4. The pre-sizing rejection path dropped initiated_by, filing a refused rule
     exit as an ordinary agent rejection.
  6. The board could not tell a rule exit from an agent close.
  7. The agent was taught a self-critique lesson from a close it never made.
  8. A failure while selecting exits could abort the agent's whole cycle.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import agent_trading_tick  # noqa: E402

from analyzing_llm_rationale import agent_trading_stats, benchmark_tools, market_data  # noqa: E402
from tests.test_agent_trading_tick import _quote  # noqa: E402
from tests.test_pre_expiry_exit_rule import NOW, _AccountCase, held, iso  # noqa: E402

RULE = agent_trading_tick.PRE_EXPIRY_EXIT_RULE


def pick(position, quote):
    return agent_trading_tick._pre_expiry_exit_candidates([position], [quote], now=NOW)


class KalshiOnlyTests(unittest.TestCase):
    """Finding 1."""

    def test_a_deep_polymarket_loss_near_close_is_left_to_the_agent(self):
        position = held(platform="polymarket", ticker="will-it-happen")
        quote = {
            "platform": "Polymarket", "ident": "will-it-happen", "question": "Q?",
            "yes_bid": 0.05, "yes_ask": 0.07, "close_time": iso(5),
        }
        self.assertEqual(pick(position, quote), [])

    def test_the_same_loss_on_kalshi_is_closed(self):
        self.assertEqual(len(pick(held(), _quote("KXHELD", bid=0.05, ask=0.07, close=iso(5)))), 1)


class BacktestedPriceSeriesTests(unittest.TestCase):
    """Finding 2: NO holders are priced at 1 - yes_ask, never the venue no_bid."""

    def test_a_no_holding_closes_on_1_minus_yes_ask_even_when_no_bid_says_otherwise(self):
        quote = _quote("KXHELD", bid=0.78, ask=0.80, close=iso(6))
        quote["no_bid"] = 0.40          # would read as only 20% down on a 0.50 entry
        chosen = pick(held(side="no", avg=0.50), quote)
        self.assertEqual(len(chosen), 1, "1 - yes_ask = 0.20 is 60% below entry")
        self.assertAlmostEqual(chosen[0]["bid"], 0.20)

    def test_a_no_holding_is_kept_when_1_minus_yes_ask_is_fine_whatever_no_bid_says(self):
        quote = _quote("KXHELD", bid=0.38, ask=0.40, close=iso(6))
        quote["no_bid"] = 0.10          # would read as 80% down
        self.assertEqual(pick(held(side="no", avg=0.50), quote), [], "1 - yes_ask = 0.60 is a gain")

    def test_the_helper_names_exactly_the_series(self):
        quote = {"yes_bid": 0.30, "yes_ask": 0.34, "no_bid": 0.05}
        self.assertEqual(agent_trading_tick._backtested_exit_price(quote, "yes"), 0.30)
        self.assertAlmostEqual(agent_trading_tick._backtested_exit_price(quote, "no"), 0.66)
        self.assertIsNone(agent_trading_tick._backtested_exit_price({"yes_bid": 0.3}, "no"))


class RuleExitsDoNotSpendTheAgentsBudgetTests(_AccountCase):
    """Finding 3."""

    def usage(self, ticker="KXHELD", side="no"):
        policy = benchmark_tools._risk_guard_policy()
        _account, usage = benchmark_tools._load_guard_account(
            self.AGENT, policy, platform="kalshi", ticker=ticker, side=side,
        )
        return usage

    def test_a_rule_exit_leaves_the_cycle_trade_count_and_cooldown_untouched(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_CYCLE_ID": "budget-cycle"}):
            self.open_yes()
            before = self.usage()
            outcomes = self.run_exits(_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5)))
            self.assertEqual([o["status"] for o in outcomes], ["closed"])
            after = self.usage()
        self.assertEqual(after["cycle_trade_count"], before["cycle_trade_count"])
        self.assertAlmostEqual(after["cycle_spend"], before["cycle_spend"])
        self.assertAlmostEqual(after["daily_risk"], before["daily_risk"])
        self.assertFalse(after["duplicate_active"], "the rule's close must not cool the market down")

    def test_an_agent_close_still_counts_as_before(self):
        """Only system-initiated rows are skipped; nothing else changes."""
        rows = [
            {"action_type": "trade", "cycle_id": "c1", "ts": "2026-09-13T12:00:00+00:00",
             "platform": "kalshi", "ticker": "KXA", "side": "no", "cash_required": 5.0, "quantity": 10,
             "metadata_json": json.dumps({"audit": {"version": 1}})},
            {"action_type": "trade", "cycle_id": "c1", "ts": "2026-09-13T12:00:00+00:00",
             "platform": "kalshi", "ticker": "KXB", "side": "no", "cash_required": 5.0, "quantity": 10,
             "metadata_json": json.dumps({"audit": {"version": 1, "initiated_by": RULE}})},
        ]
        policy = benchmark_tools._risk_guard_policy()
        policy = type(policy)(**{**policy.__dict__, "cycle_id": "c1"})
        usage = benchmark_tools._risk_usage(rows, policy=policy, platform="kalshi", ticker="KXZ", side="yes")
        self.assertEqual(usage["cycle_trade_count"], 1)
        self.assertAlmostEqual(usage["cycle_spend"], 5.0)

    def test_unreadable_metadata_is_treated_as_the_agent(self):
        self.assertIsNone(benchmark_tools._audit_initiated_by("not json"))
        self.assertIsNone(benchmark_tools._audit_initiated_by(None))
        self.assertIsNone(benchmark_tools._audit_initiated_by({"audit": "x"}))
        self.assertEqual(benchmark_tools._audit_initiated_by({"audit": {"initiated_by": RULE}}), RULE)


class PreSizingRejectionTagTests(_AccountCase):
    """Finding 4: the third write path records the tag too."""

    def test_a_rule_close_refused_before_sizing_is_filed_as_a_rule_exit(self):
        self.open_yes()
        ctx = benchmark_tools.ToolContext(agent_id=self.AGENT, require_kelly_sizing=True, initiated_by=RULE)
        # yes_bid 0 makes no_ask = 1.0, which place_trade refuses before sizing.
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.0, ask=0.02)):
            result = benchmark_tools.place_trade(
                {"platform": "kalshi", "ticker": "KXHELD", "side": "no", "sizing_mode": "close"}, ctx,
            )
        self.assertFalse(result.get("ok"))
        rejections = [a for a in self.audits() if a.get("status") == "rejected_before_sizing"]
        self.assertEqual(len(rejections), 1, self.audits())
        self.assertEqual(rejections[0].get("initiated_by"), RULE)

    def test_an_agent_order_refused_before_sizing_stays_untagged(self):
        self.open_yes()
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.0, ask=0.02)):
            benchmark_tools.place_trade(
                {"platform": "kalshi", "ticker": "KXHELD", "side": "no", "sizing_mode": "close"},
                benchmark_tools.ToolContext(agent_id=self.AGENT, require_kelly_sizing=True),
            )
        rejections = [a for a in self.audits() if a.get("status") == "rejected_before_sizing"]
        self.assertEqual([r.get("initiated_by") for r in rejections], [None])


class BoardShowsRuleExitsTests(_AccountCase):
    """Finding 6: counted in the score, and labelled."""

    def test_the_leaderboard_counts_rule_exits_and_still_scores_them(self):
        self.open_yes()
        self.run_exits(_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5)))
        with benchmark_tools._account_transaction() as conn:
            row = next(r for r in agent_trading_stats.compute_agent_leaderboard(conn, {}) if r["agent_id"] == self.AGENT)
        self.assertEqual(row["rule_exit_count"], 1)
        self.assertEqual(row["realized_count"], 1, "the loss is the account's and stays in the score")

    def test_an_agent_close_is_not_counted_as_a_rule_exit(self):
        self.open_yes()
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.10, ask=0.12)):
            benchmark_tools.place_trade(
                {"platform": "kalshi", "ticker": "KXHELD", "side": "no", "sizing_mode": "close"},
                benchmark_tools.ToolContext(agent_id=self.AGENT, require_kelly_sizing=True),
            )
        with benchmark_tools._account_transaction() as conn:
            row = next(r for r in agent_trading_stats.compute_agent_leaderboard(conn, {}) if r["agent_id"] == self.AGENT)
        self.assertEqual(row["rule_exit_count"], 0)

    def test_each_activity_item_says_who_placed_it(self):
        tagged = json.dumps({"audit": {"version": 1, "initiated_by": RULE}})
        untagged = json.dumps({"audit": {"version": 1}})
        self.assertEqual(agent_trading_stats.fill_context(tagged, 10)["initiated_by"], RULE)
        self.assertNotIn("initiated_by", agent_trading_stats.fill_context(untagged, 10))


class LearningLessonTests(_AccountCase):
    """Finding 7."""

    def test_a_rule_exit_is_not_framed_as_the_agents_own_close(self):
        lesson = agent_trading_tick._learning_lesson("trade", -25.0, RULE)
        self.assertIn("closed this position, not you", lesson)
        self.assertNotIn("The position close lost money", lesson)

    def test_an_agent_close_keeps_its_lesson(self):
        self.assertIn("The position close lost money", agent_trading_tick._learning_lesson("trade", -25.0))

    def test_the_stored_lesson_for_a_rule_exit_is_the_rule_wording(self):
        self.open_yes()
        self.run_exits(_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5)))
        with benchmark_tools._account_transaction() as conn:
            agent_trading_tick._refresh_learning(conn, self.AGENT)
            lessons = [r["lesson"] for r in conn.execute(
                "SELECT lesson FROM agent_learning WHERE agent_id = ?", (self.AGENT,),
            )]
        self.assertEqual(len(lessons), 1)
        self.assertIn("closed this position, not you", lessons[0])


class NeverAbortTheCycleTests(_AccountCase):
    """Finding 8, and the backend guard that replaces finding 5 for this rule."""

    def test_a_failure_while_selecting_exits_means_no_exits_not_a_crash(self):
        self.open_yes()
        with (
            mock.patch.object(benchmark_tools, "_account_summary", side_effect=RuntimeError("bad row")),
            self.assertLogs(agent_trading_tick.logger.name, "WARNING"),
        ):
            outcomes = agent_trading_tick._run_pre_expiry_exits(
                self.AGENT, [_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))], now=NOW,
            )
        self.assertEqual(outcomes, [])

    def test_without_a_sqlite_store_the_rule_does_not_run(self):
        """Reading SQLite positions while place_trade trades Datastore would mix two books.

        Position loading is made to succeed with a qualifying position, so the
        only thing standing between the rule and a trade is the backend guard.
        Without that, a missing store would raise and the cycle-safety handler
        would return [] anyway -- passing this test for the wrong reason.
        """
        import contextlib

        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch.object(benchmark_tools, "_account_transaction", lambda: contextlib.nullcontext(None)),
            mock.patch.object(benchmark_tools, "_account_summary", return_value={"open_positions": [held()]}),
            mock.patch.object(benchmark_tools, "place_trade", return_value={"ok": False}) as trade,
        ):
            os.environ.pop("FORESEA_AGENT_ACCOUNT_DB_PATH", None)
            outcomes = agent_trading_tick._run_pre_expiry_exits(
                self.AGENT, [_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))], now=NOW,
            )
        self.assertEqual(outcomes, [])
        trade.assert_not_called()

    def test_with_a_sqlite_store_the_same_setup_does_trade(self):
        """The control: the guard is the difference, not the mocks."""
        import contextlib

        with (
            mock.patch.object(benchmark_tools, "_account_transaction", lambda: contextlib.nullcontext(None)),
            mock.patch.object(benchmark_tools, "_account_summary", return_value={"open_positions": [held()]}),
            mock.patch.object(benchmark_tools, "place_trade", return_value={"ok": False}) as trade,
        ):
            agent_trading_tick._run_pre_expiry_exits(
                self.AGENT, [_quote("KXHELD", bid=0.10, ask=0.12, close=iso(5))], now=NOW,
            )
        trade.assert_called_once()


if __name__ == "__main__":
    unittest.main()
