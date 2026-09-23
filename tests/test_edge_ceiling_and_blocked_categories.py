"""Two entry gates drawn from what the agents' own book and track record say.

Every account is down (-$8.1k across eight agents), and the losses are not
spread evenly:

  - The stated edge is inverted. The published track record, 4,800 resolved
    forecasts, scores the model against the market at roughly nil below 5pp
    of disagreement, -0.015 at 5-10pp, -0.028 at 10-20pp and -0.066 at 20pp+.
    The agents' book agrees in money: the 20pp+ bucket returned -34% on $9.5k
    staked. Sizing scales the stake with the stated edge, so the largest
    positions are the least credible ones. New exposure now stops at 20pp.
  - Crypto carried most of the rest: -86% across the agents' book, and the
    one domain besides geopolitics the published track record scores
    negative on its own. No new exposure is opened there. The gate takes
    any category name, so weather -- 55 markets, -$2.5k, -52% -- can be
    added by configuration; it stays tradable here because its research
    path is a feature in its own right.

Both gates are entry-only. An agent can always close what it holds, which is
what keeps a blocked category from trapping a position.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyzing_llm_rationale import benchmark_tools, market_data  # noqa: E402

AGENT = "gate-model"


def quote(ticker, *, bid=0.40, ask=0.42, question="Will the Yankees win the World Series?", category=None):
    q = {
        "platform": "Kalshi", "ident": ticker, "question": question,
        "probability": (bid + ask) / 2, "yes_bid": bid, "yes_ask": ask,
        "close_time": "2027-01-01T00:00:00Z", "created_time": "2026-01-01T00:00:00Z",
    }
    if category is not None:
        q["category"] = category
    return q


class _GateCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = mock.patch.dict(os.environ, {
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(tmp.name) / "accounts.sqlite"),
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(tmp.name) / "ledger.jsonl"),
            "FORESEA_AGENT_NOTES_PATH": str(Path(tmp.name) / "notes.json"),
            "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
            "FORESEA_AGENT_CYCLE_ID": "gate-cycle",
            # Generous size limits: these tests are about the two gates, and a
            # concentration or spend rejection would mask the reason under test.
            "FORESEA_AGENT_CONCENTRATION_LIMIT": "1.0",
            "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "10",
            "FORESEA_MAX_ORDER_NOTIONAL": "100000",
        }, clear=False)
        env.start()
        self.addCleanup(env.stop)
        resolve = mock.patch.object(market_data, "resolve_kalshi", return_value=None)
        resolve.start()
        self.addCleanup(resolve.stop)

    def trade(self, ticker, *, model_probability=None, side="yes", sizing_mode="quarter_kelly", q=None, **fields):
        # Strict mode (the scheduled tick) refuses a manual quantity, so every
        # entry here is sized the way a real cycle sizes one.
        args = {"ticker": ticker, "side": side, "price": 0.42, "sizing_mode": sizing_mode}
        if model_probability is not None:
            args["model_probability"] = model_probability
        args.update(fields)
        with mock.patch.object(market_data, "fetch_kalshi", return_value=q or quote(ticker)):
            return benchmark_tools.place_trade(
                args, benchmark_tools.ToolContext(agent_id=AGENT, require_kelly_sizing=True),
            )

    def reasons(self, result):
        return (result.get("risk_guard") or {}).get("reasons") or []


class EdgeCeilingTests(_GateCase):
    def test_an_entry_claiming_20pp_of_edge_is_refused(self):
        result = self.trade("KXEDGE", model_probability=0.75)   # 0.75 - 0.42 = 33pp
        self.assertFalse(result["ok"], result)
        self.assertIn("edge_beyond_credible_range", self.reasons(result))

    def test_an_entry_inside_the_ceiling_still_trades(self):
        result = self.trade("KXEDGEOK", model_probability=0.55)  # 13pp
        self.assertTrue(result["ok"], result)
        self.assertNotIn("edge_beyond_credible_range", self.reasons(result))

    def test_the_ceiling_is_exactly_where_the_record_turns(self):
        just_under = self.trade("KXEDGEUNDER", model_probability=0.619)  # 19.9pp
        just_over = self.trade("KXEDGEOVER", model_probability=0.621)    # 20.1pp
        self.assertTrue(just_under["ok"], just_under)
        self.assertFalse(just_over["ok"], just_over)

    def test_the_ceiling_can_be_lifted(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_MAX_CREDIBLE_EDGE": "0"}):
            result = self.trade("KXEDGEOFF", model_probability=0.75)
        self.assertTrue(result["ok"], result)

    def test_the_audit_records_the_ceiling_each_order_was_held_to(self):
        self.trade("KXEDGEAUDIT", model_probability=0.55)
        with benchmark_tools._account_transaction() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM agent_actions WHERE agent_id = ? AND ticker = ?",
                (AGENT, "KXEDGEAUDIT"),
            ).fetchone()
        risk = json.loads(row["metadata_json"])["audit"]["risk"]
        self.assertEqual(risk["max_credible_edge"], 0.20)
        self.assertIsNone(risk["blocked_category"])


class BlockedCategoryTests(_GateCase):
    def test_weather_is_tradable_by_default(self):
        """Blocked by configuration, never by default: the lane stays open."""
        result = self.trade("KXHIGHNY-26SEP22", model_probability=0.55)
        self.assertTrue(result["ok"], result)

    def test_weather_can_be_added_to_the_block_list(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_BLOCKED_CATEGORIES": "crypto,weather"}):
            result = self.trade("KXHIGHNY-26SEP23", model_probability=0.55)
        self.assertFalse(result["ok"], result)
        self.assertIn("blocked_category_weather", self.reasons(result))

    def test_a_crypto_market_is_not_opened(self):
        q = quote("KXBTCPRICE", question="Will bitcoin close above $100k this year?")
        result = self.trade("KXBTCPRICE", model_probability=0.55, q=q)
        self.assertFalse(result["ok"], result)
        self.assertIn("blocked_category_crypto", self.reasons(result))

    def test_an_ordinary_market_is_untouched(self):
        result = self.trade("KXSPORTS", model_probability=0.55)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.reasons(result), [])

    def test_a_held_blocked_position_can_still_be_closed(self):
        """The gate must never trap an agent in a market it is already in."""
        btc = quote("KXBTCHELD", question="Will bitcoin close above $100k?")
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_BLOCKED_CATEGORIES": ""}):
            opened = self.trade("KXBTCHELD", model_probability=0.55, q=btc)
        self.assertTrue(opened["ok"], opened)

        closed = self.trade("KXBTCHELD", side="no", sizing_mode="close", q=btc)
        self.assertTrue(closed["ok"], closed)
        self.assertNotIn("blocked_category_crypto", self.reasons(closed))

    def test_the_block_list_can_be_emptied(self):
        btc = quote("KXBTCOPEN", question="Will bitcoin close above $100k?")
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_BLOCKED_CATEGORIES": ""}):
            result = self.trade("KXBTCOPEN", model_probability=0.55, q=btc)
        self.assertTrue(result["ok"], result)

    def test_the_block_list_takes_other_categories(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_BLOCKED_CATEGORIES": "politics"}):
            blocked = self.trade(
                "KXPRES", model_probability=0.55,
                q=quote("KXPRES", question="Will the president sign the bill?"),
            )
            crypto_now_allowed = self.trade(
                "KXBTCFREE", model_probability=0.55,
                q=quote("KXBTCFREE", question="Will bitcoin close above $100k?"),
            )
        self.assertIn("blocked_category_politics", self.reasons(blocked))
        self.assertTrue(crypto_now_allowed["ok"], crypto_now_allowed)


class CandidateDiscoveryTests(unittest.TestCase):
    """Blocked markets are also kept off the candidate list, so no cycle researches them."""

    def test_blocked_candidates_are_dropped_and_the_rest_kept(self):
        import agent_trading_tick

        quotes = [
            quote("KXBTC", question="Will bitcoin close above $100k?"),
            quote("KXHIGHNY-26SEP22"),
            quote("KXWS", question="Will the Yankees win the World Series?"),
        ]
        kept = agent_trading_tick._drop_blocked_candidates(quotes)
        self.assertEqual([q["ident"] for q in kept], ["KXHIGHNY-26SEP22", "KXWS"])

    def test_the_weather_lane_runs_while_weather_is_tradable(self):
        import agent_trading_tick

        with mock.patch.object(market_data, "list_kalshi", return_value=[]) as listed:
            agent_trading_tick._discover_weather_candidates(set(), limit=3)
        self.assertTrue(listed.called)

    def test_the_weather_lane_is_skipped_once_weather_is_blocked(self):
        """Blocking a category must not leave its lane spending venue and NWS calls."""
        import agent_trading_tick

        with (
            mock.patch.dict(os.environ, {"FORESEA_AGENT_BLOCKED_CATEGORIES": "crypto,weather"}),
            mock.patch.object(market_data, "list_kalshi", side_effect=AssertionError("must not call the venue")),
        ):
            self.assertEqual(agent_trading_tick._discover_weather_candidates(set(), limit=3), [])


if __name__ == "__main__":
    unittest.main()
