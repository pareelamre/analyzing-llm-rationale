import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import scripts.agent_trading_tick as tick

from analyzing_llm_rationale import agent_capabilities as ac
from analyzing_llm_rationale import benchmark_tools, market_data


def _fetch_kalshi_quotes(quotes):
    def _fetch(ticker):
        spec = quotes.get(ticker)
        if spec is None:
            raise market_data.MarketDataError(f"no mock quote configured for {ticker}")
        if isinstance(spec, (int, float)):
            return {"yes_ask": spec, "no_ask": spec}
        return dict(spec)
    return _fetch


class TestBidAskSpreadGuard(unittest.TestCase):
    def test_spread_guard_rejects_wide_absolute_spread(self):
        """A spread > 12c (e.g. ask=0.60, bid=0.45 -> spread 0.15) must be rejected."""
        ctx = benchmark_tools.ToolContext(agent_id="durability-test-agent", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({
                        "KXWIDE": {"yes_ask": 0.60, "yes_bid": 0.45, "no_ask": 0.55, "no_bid": 0.40}
                    }),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {
                        "ticker": "KXWIDE", "side": "yes", "price": 0.60,
                        "sizing_mode": "quarter_kelly", "model_probability": 0.80,
                    },
                    ctx,
                )

        self.assertFalse(result["ok"])
        self.assertIn("wide_bid_ask_spread", result["risk_guard"]["reasons"])
        spread_check = result["risk_guard"].get("spread_check")
        self.assertIsNotNone(spread_check)
        self.assertFalse(spread_check["clears"])
        self.assertEqual(spread_check["spread"], 0.15)

    def test_spread_guard_rejects_wide_relative_spread(self):
        """A spread ratio > 25% (e.g. ask=0.20, bid=0.12 -> spread 0.08, ratio 40%) must be rejected."""
        ctx = benchmark_tools.ToolContext(agent_id="durability-test-agent", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({
                        "KXRATIO": {"yes_ask": 0.20, "yes_bid": 0.12, "no_ask": 0.88, "no_bid": 0.80}
                    }),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {
                        "ticker": "KXRATIO", "side": "yes", "price": 0.20,
                        "sizing_mode": "quarter_kelly", "model_probability": 0.35,
                    },
                    ctx,
                )

        self.assertFalse(result["ok"])
        self.assertIn("wide_bid_ask_spread", result["risk_guard"]["reasons"])
        spread_check = result["risk_guard"]["spread_check"]
        self.assertFalse(spread_check["clears"])
        self.assertEqual(spread_check["spread_ratio"], 0.4)

    def test_spread_guard_allows_tight_spread(self):
        """A tight spread (e.g. ask=0.52, bid=0.50 -> spread 0.02, ratio 3.8%) passes."""
        ctx = benchmark_tools.ToolContext(agent_id="durability-test-agent", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
                "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            }
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({
                        "KXTIGHT": {"yes_ask": 0.52, "yes_bid": 0.50, "no_ask": 0.50, "no_bid": 0.48}
                    }),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {
                        "ticker": "KXTIGHT", "side": "yes", "price": 0.52,
                        "sizing_mode": "quarter_kelly", "model_probability": 0.70,
                    },
                    ctx,
                )

        self.assertTrue(result["ok"])
        self.assertTrue(result["risk_guard"]["allowed"])
        self.assertTrue(result["risk_guard"]["spread_check"]["clears"])

    def test_spread_guard_exempts_position_closing(self):
        """Closing an existing position must succeed even if the spread is wide (risk-reducing exemption)."""
        ctx = benchmark_tools.ToolContext(agent_id="durability-test-agent", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            db_path = str(Path(td) / "accounts.sqlite")
            ledger_path = str(Path(td) / "ledger.jsonl")
            env = {
                "FORESEA_AGENT_TOOL_LEDGER_PATH": ledger_path,
                "FORESEA_AGENT_ACCOUNT_DB_PATH": db_path,
                "FORESEA_MAX_ORDER_NOTIONAL": "1000",
                # The opening trade is setup for the close under test, and its
                # stated edge sits on the credible-edge ceiling.
                "FORESEA_AGENT_MAX_CREDIBLE_EDGE": "0",
            }
            # First open a position under tight spread
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({
                        "KXCLOSE": {"yes_ask": 0.50, "yes_bid": 0.49, "no_ask": 0.51, "no_bid": 0.50}
                    }),
                ),
            ):
                open_res = benchmark_tools.place_trade(
                    {
                        "ticker": "KXCLOSE", "side": "yes", "price": 0.50,
                        "sizing_mode": "quarter_kelly", "model_probability": 0.70,
                    },
                    ctx,
                )
                self.assertTrue(open_res["ok"])

            # Now close the position under wide spread (ask 0.60, bid 0.40 -> spread 0.20)
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=_fetch_kalshi_quotes({
                        "KXCLOSE": {"yes_ask": 0.60, "yes_bid": 0.40, "no_ask": 0.60, "no_bid": 0.40}
                    }),
                ),
            ):
                close_res = benchmark_tools.place_trade(
                    {"ticker": "KXCLOSE", "side": "no", "price": 0.60, "sizing_mode": "close"},
                    ctx,
                )
                self.assertTrue(close_res["ok"])
                self.assertNotIn("wide_bid_ask_spread", close_res["risk_guard"].get("reasons", []))


class TestTailContractAsymmetryProtection(unittest.TestCase):
    def test_expensive_contract_shrinkage_and_cap(self):
        """Contracts trading >= 0.85 must elevate shrinkage to >= 40% (even from 15%) and cap position."""
        args = {"sizing_mode": "convex_conviction", "model_probability": 0.96}
        plan = benchmark_tools._sizing_plan(
            args,
            price=0.88,
            side="yes",
            account_value=1000.0,
        )
        self.assertTrue(plan["eligible"])
        # Convex conviction base shrinkage is 0.15; for >= 0.85 it must elevate to at least 0.40
        self.assertGreaterEqual(plan["market_shrinkage"], 0.40)
        self.assertLessEqual(plan["max_position_fraction"], 0.15)
        self.assertLessEqual(plan["target_fraction"], 0.15 + 1e-6)

    def test_cheap_contract_tail_cap(self):
        """Contracts trading <= 0.10 must be capped at <= 10% account value."""
        args = {"sizing_mode": "quarter_kelly", "model_probability": 0.20}
        plan = benchmark_tools._sizing_plan(
            args,
            price=0.08,
            side="yes",
            account_value=1000.0,
        )
        self.assertTrue(plan["eligible"])
        self.assertLessEqual(plan["max_position_fraction"], 0.10)
        self.assertLessEqual(plan["target_fraction"], 0.10 + 1e-6)

    def test_expensive_contract_requires_at_least_3pp_edge(self):
        """Contracts >= 0.85 require at least 3 percentage points of edge."""
        # Edge is 0.89 - 0.88 = 0.01 (< 0.03 hurdle)
        args = {"sizing_mode": "quarter_kelly", "model_probability": 0.89}
        plan = benchmark_tools._sizing_plan(
            args,
            price=0.88,
            side="yes",
            account_value=1000.0,
        )
        self.assertFalse(plan["eligible"])
        self.assertEqual(plan["min_edge"], 0.03)
        self.assertEqual(plan["reason"], "edge_below_threshold")


class TestEarlyProfitHarvesting(unittest.TestCase):
    def test_candidates_selection_qualifies_winning_positions(self):
        """Positions capturing >= 85% of profit with bid >= 0.85 qualify for early harvest."""
        positions = [
            # Bought at 0.20, live bid is 0.90 -> max profit is 0.80, captured 0.70 (87.5% >= 85%), bid 0.90 >= 0.85 -> QUALIFIES
            {
                "platform": "kalshi",
                "ticker": "KXWINNER",
                "side": "yes",
                "quantity": 10.0,
                "avg_entry_price": 0.20,
            },
            # Bought at 0.50, live bid is 0.70 -> max profit is 0.50, captured 0.20 (40% < 85%) -> REJECTED
            {
                "platform": "kalshi",
                "ticker": "KXMIDS",
                "side": "yes",
                "quantity": 5.0,
                "avg_entry_price": 0.50,
            },
            # Bought at 0.10, live bid is 0.80 -> captured 0.70 / 0.90 = 77.8% < 85%, bid < 0.85 -> REJECTED
            {
                "platform": "kalshi",
                "ticker": "KXNOTYET",
                "side": "yes",
                "quantity": 5.0,
                "avg_entry_price": 0.10,
            },
        ]
        now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
        close_time = (now + timedelta(days=2)).isoformat()
        held_quotes = [
            {"ident": "KXWINNER", "platform": "kalshi", "yes_bid": 0.90, "close_time": close_time},
            {"ident": "KXMIDS", "platform": "kalshi", "yes_bid": 0.70, "close_time": close_time},
            {"ident": "KXNOTYET", "platform": "kalshi", "yes_bid": 0.80, "close_time": close_time},
        ]

        candidates = tick._early_profit_harvest_candidates(positions, held_quotes, now=now)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["ticker"], "KXWINNER")
        self.assertAlmostEqual(candidates[0]["profit_captured_pct"], 0.875)

    def test_candidates_selection_ignores_closed_markets(self):
        """Positions whose market close time is in the past are ignored."""
        now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
        past_close = (now - timedelta(hours=1)).isoformat()
        positions = [
            {
                "platform": "kalshi",
                "ticker": "KXPAST",
                "side": "yes",
                "quantity": 10.0,
                "avg_entry_price": 0.20,
            }
        ]
        held_quotes = [
            {"ident": "KXPAST", "platform": "kalshi", "yes_bid": 0.92, "close_time": past_close}
        ]
        candidates = tick._early_profit_harvest_candidates(positions, held_quotes, now=now)
        self.assertEqual(len(candidates), 0)

    def test_learning_lesson_early_harvest(self):
        """The learning lesson correctly describes early profit harvest rule."""
        lesson = tick._learning_lesson("position_close", 7.0, initiated_by=tick.EARLY_HARVEST_RULE)
        self.assertIn("early profit harvest rule", lesson)
        self.assertIn(">= 85%", lesson)


class TestSystemPromptStandards(unittest.TestCase):
    def test_prompt_includes_durability_standards(self):
        prompt = ac.build_system_prompt([{"name": "forecast", "description": "d"}], 4)
        self.assertIn("Microstructure & Spread Discipline", prompt)
        self.assertIn("Tail Risk & Variance (Steamroller Protection)", prompt)
        self.assertIn("Resolution Ambiguity Screening", prompt)
        self.assertIn("Profit Harvesting & De-risking", prompt)


if __name__ == "__main__":
    unittest.main()
