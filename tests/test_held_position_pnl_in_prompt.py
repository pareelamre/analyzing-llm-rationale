"""Tell each agent what its open positions are worth now, and how long they have.

Positions held to settlement won 4 of 26 and lost $3,984 -- 86% of the
fleet's realized losses -- so the numbers that matter for exiting a loser are
the current value of the position, its unrealized P&L, and the time left
before it resolves.

The tick already re-quoted every held market, but it showed that quote as a
raw YES and NO book in the candidates block, while the entry price and cost
basis sat in the portfolio block. To tell whether a position was losing, an
agent had to join the two blocks itself, pick the right side's bid (the NO
bid for a NO holding), multiply by quantity, subtract cost basis, and work
out hours to close from two timestamps. Each held position now states the
result on one line.

A backtest of mechanical exit rules on real price paths did not justify
forcing exits: fixed and trailing stop-losses cut eventual winners on these
short-dated, news-driven markets and did no better than the agents. This
change adds information and forces nothing.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
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


def position(side="YES", quantity=100.0, avg=0.40, ticker="KXHELD"):
    return {
        "platform": "kalshi", "ticker": ticker, "side": side,
        "quantity": quantity, "avg_entry_price": avg, "cost_basis": quantity * avg,
    }


def line(pos, quote):
    return agent_trading_tick._fmt_open_position(pos, quote, now=NOW)


class OpenPositionLineTests(unittest.TestCase):
    def test_a_losing_yes_position_states_its_value_and_loss(self):
        text = line(position(), _quote("KXHELD", bid=0.10, ask=0.12, close="2026-09-14T06:00:00Z"))
        self.assertIn("exit value now 0.10", text)
        self.assertIn("unrealized -$30.00 (-75%) before fees", text)
        self.assertIn("closes in 18h", text)

    def test_a_no_holding_is_valued_at_the_no_bid_not_the_yes_bid(self):
        """The mistake an agent reading the raw YES book would make."""
        quote = _quote("KXHELD", bid=0.60, ask=0.70, close="2026-09-14T06:00:00Z")
        text = line(position(side="NO", avg=0.50), quote)
        # NO bid = 1 - YES ask = 0.30, so 100 contracts are worth $30 against a $50 basis.
        self.assertIn("exit value now 0.30", text)
        self.assertIn("unrealized -$20.00 (-40%)", text)
        self.assertNotIn("exit value now 0.60", text)

    def test_a_winning_position_is_shown_as_a_gain(self):
        text = line(position(avg=0.20), _quote("KXHELD", bid=0.55, ask=0.57, close="2026-09-20T12:00:00Z"))
        self.assertIn("unrealized +$35.00 (+175%)", text)
        self.assertIn("closes in 7d", text)

    def test_a_position_without_a_quote_says_its_value_is_unknown(self):
        text = line(position(), None)
        self.assertIn("no live quote this cycle", text)
        self.assertNotIn("unrealized", text)

    def test_a_position_with_no_bid_says_it_cannot_be_exited_at_a_price(self):
        quote = _quote("KXHELD", bid=0.0, ask=0.03, close="2026-09-14T06:00:00Z")
        text = line(position(), quote)
        self.assertIn("no executable YES bid right now", text)
        self.assertNotIn("unrealized", text)

    def test_a_market_past_its_close_is_awaiting_settlement(self):
        text = line(position(), _quote("KXHELD", bid=0.05, ask=0.07, close="2026-09-12T00:00:00Z"))
        self.assertIn("past its close time, awaiting settlement", text)

    def test_an_unparseable_close_time_is_stated_not_guessed(self):
        text = line(position(), _quote("KXHELD", bid=0.05, ask=0.07, close="soon"))
        self.assertIn("close time unknown", text)

    def test_the_entry_details_are_still_there(self):
        text = line(position(), _quote("KXHELD", bid=0.10, ask=0.12))
        self.assertIn("KXHELD YES: 100.0 contracts, avg entry 0.40, cost basis", text)


class HoursUntilTests(unittest.TestCase):
    def test_hours_until_and_the_horizon_bucket_agree(self):
        """One parser now feeds both, so they cannot disagree about a close time."""
        close = "2026-09-14T12:00:00Z"
        self.assertAlmostEqual(agent_trading_tick._hours_until(close, now=NOW), 24.0)
        self.assertEqual(agent_trading_tick._paper_horizon_bucket(close, now=NOW), "short")
        self.assertEqual(agent_trading_tick._paper_horizon_bucket(None, now=NOW), "unknown")
        self.assertEqual(agent_trading_tick._paper_horizon_bucket("nope", now=NOW), "unknown")


class PortfolioBlockTests(unittest.TestCase):
    def _open_position(self, td):
        os.environ["FORESEA_AGENT_ACCOUNT_DB_PATH"] = str(Path(td) / "accounts.sqlite")
        ctx = benchmark_tools.ToolContext(agent_id="model-pnl")
        with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.40, ask=0.42)):
            benchmark_tools.place_trade({"ticker": "KXHELD", "side": "yes", "price": 0.42, "quantity": 10}, ctx)

    def test_the_block_shows_each_held_position_with_its_live_value(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=False):
            self._open_position(td)
            with benchmark_tools._account_transaction() as conn:
                block = agent_trading_tick._build_portfolio_block(
                    conn, "model-pnl", None,
                    held_quotes=[_quote("KXHELD", bid=0.20, ask=0.22, close="2026-09-14T06:00:00Z")],
                    now=NOW,
                )
        self.assertIn("KXHELD yes", block)
        self.assertIn("exit value now 0.20", block)
        self.assertIn("unrealized -$", block)

    def test_a_caller_without_quotes_gets_the_plain_line(self):
        """No quotes passed is not the same claim as every position unquoted."""
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=False):
            self._open_position(td)
            with benchmark_tools._account_transaction() as conn:
                block = agent_trading_tick._build_portfolio_block(conn, "model-pnl", None)
        self.assertIn("KXHELD", block)
        self.assertNotIn("no live quote this cycle", block)
        self.assertNotIn("unrealized", block)


class RunCycleTests(unittest.TestCase):
    def test_the_agent_prompt_carries_the_held_position_value(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json"),
                "FORESEA_AGENT_CYCLE_ID": "pnl-cycle",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                ctx = benchmark_tools.ToolContext(agent_id="model-pnl-cycle")
                with mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.40, ask=0.42)):
                    benchmark_tools.place_trade({"ticker": "KXHELD", "side": "yes", "price": 0.42, "quantity": 10}, ctx)
                with (
                    mock.patch.object(agent_trading_tick, "_init_local_agent"),
                    mock.patch.object(market_data, "fetch_kalshi", return_value=_quote("KXHELD", bid=0.05, ask=0.07)),
                    mock.patch.object(market_data, "resolve_kalshi", return_value=None),
                    mock.patch.object(market_data, "list_kalshi", return_value=[_quote("KXNEW")]),
                    mock.patch.object(market_data, "list_polymarket", return_value=[]),
                    mock.patch.object(
                        agent_trading_tick, "_call_agent_analyze",
                        return_value=SimpleNamespace(thesis="Held.", tool_transcript=[]),
                    ) as call_mock,
                ):
                    agent_trading_tick.run_cycle("model-pnl-cycle")
        question = call_mock.call_args[0][0]
        self.assertIn("KXHELD", question)
        self.assertIn("exit value now 0.05", question)
        self.assertIn("unrealized -$", question)


if __name__ == "__main__":
    unittest.main()
