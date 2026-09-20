import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import agent_trading_tick

from analyzing_llm_rationale import benchmark_tools, trading
from analyzing_llm_rationale.benchmark_tools import (
    _extract_market_cluster,
    place_trade,
)


def _quote(
    ticker: str,
    *,
    bid: float = 0.19,
    ask: float = 0.20,
    prob: float = 0.20,
    lead_days: float = 14.0,
    category: str = "politics",
    resolution_criteria: str = "Official exchange resolution criteria.",
):
    return {
        "ticker": ticker,
        "platform": "kalshi",
        "marketable": True,
        "status": "active",
        "yes_ask": ask,
        "yes_bid": bid,
        "no_ask": round(1.0 - bid, 4),
        "no_bid": round(1.0 - ask, 4),
        "price": ask,
        "real_ask": ask,
        "observed_ask": ask,
        "observed_bid": bid,
        "spread_check": {"clears": True, "spread": ask - bid, "spread_ratio": (ask - bid) / ask, "status": "ok"},
        "lead_days": lead_days,
        "category": category,
        "resolution_criteria": resolution_criteria,
    }


def _book():
    return {
        "orderbook_fp": {
            "no_dollars": [["0.80", "10000"]],
            "yes_dollars": [["0.20", "10000"]],
        }
    }


class MarketClusterExtractionTests(unittest.TestCase):
    def test_kalshi_strike_ladders_share_cluster(self):
        c1 = _extract_market_cluster("KXHIGHNY-26SEP20-T71", platform="kalshi")
        c2 = _extract_market_cluster("KXHIGHNY-26SEP20-T72", platform="kalshi")
        c3 = _extract_market_cluster("KXHIGHNY-26SEP20-B70", platform="kalshi")
        self.assertEqual(c1, "kxhighny-26sep20")
        self.assertEqual(c2, "kxhighny-26sep20")
        self.assertEqual(c3, "kxhighny-26sep20")

    def test_polymarket_multi_date_ceasefire_shares_cluster(self):
        c1 = _extract_market_cluster("us-x-iran-ceasefire-through-september-25-2026", platform="polymarket")
        c2 = _extract_market_cluster("us-x-iran-ceasefire-through-september-30-2026", platform="polymarket")
        self.assertEqual(c1, "us-x-iran-ceasefire")
        self.assertEqual(c2, "us-x-iran-ceasefire")

    def test_polymarket_interest_rate_dates_share_cluster(self):
        c1 = _extract_market_cluster("fed-interest-rate-cut-in-september-2026", platform="polymarket")
        self.assertEqual(c1, "fed-interest-rate-cut")


class ClusterConcentrationGuardTests(unittest.TestCase):
    def _base_env(self, td: str) -> dict:
        return {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            "FORESEA_AGENT_ACCOUNT_VALUE": "10000",
            "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.15",
            "FORESEA_AGENT_CLUSTER_CONCENTRATION_LIMIT": "0.06",  # 6% = $600 max per cluster
            "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_DAILY_RISK_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_MAX_OPEN_MARKETS": "20",
            "FORESEA_AGENT_MAX_TRADES_PER_CYCLE": "10",
            "FORESEA_MAX_ORDER_NOTIONAL": "2000.0",
        }

    def test_blocks_stacked_orders_exceeding_cluster_cap(self):
        ctx = benchmark_tools.ToolContext(agent_id="qwen3-8-27b", require_kelly_sizing=False)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=lambda t: _quote(t, bid=0.19, ask=0.20, prob=0.20, lead_days=14.0),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                # Order 1: Buy $200 of strike T71 (cluster: kxhighny-26sep20). 1000 shares * 0.20 = $200.
                first = place_trade(
                    {
                        "ticker": "KXHIGHNY-26SEP20-T71",
                        "side": "yes",
                        "price": 0.20,
                        "quantity": 1000,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(first["ok"], f"First order failed: {first}")

                # Order 2: Buy $200 of strike T72 (same cluster). Total now $400 <= $600 cap.
                second = place_trade(
                    {
                        "ticker": "KXHIGHNY-26SEP20-T72",
                        "side": "yes",
                        "price": 0.20,
                        "quantity": 1000,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(second["ok"], f"Second order failed: {second}")

                # Order 3: Attempt to buy $300 of strike B70 (same cluster). Total would reach $700 > $600 cap.
                third = place_trade(
                    {
                        "ticker": "KXHIGHNY-26SEP20-B70",
                        "side": "yes",
                        "price": 0.20,
                        "quantity": 1500,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertFalse(third["ok"])
                self.assertEqual(third["reason"], "cluster_concentration_limit")
                self.assertIn("cluster_concentration_limit", third["message"])

    def test_close_is_exempt_from_cluster_concentration_cap(self):
        ctx = benchmark_tools.ToolContext(agent_id="qwen3-8-27b", require_kelly_sizing=False)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=lambda t: _quote(t, bid=0.19, ask=0.20, prob=0.20, lead_days=14.0),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                first = place_trade(
                    {
                        "ticker": "KXHIGHNY-26SEP20-T71",
                        "side": "yes",
                        "price": 0.20,
                        "quantity": 1000,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(first["ok"], f"First order failed: {first}")

                # Closing the position should succeed even if cluster limits are set to 0.01
                with mock.patch.dict(os.environ, {"FORESEA_AGENT_CLUSTER_CONCENTRATION_LIMIT": "0.01"}):
                    close_res = place_trade(
                        {
                            "ticker": "KXHIGHNY-26SEP20-T71",
                            "side": "no",
                            "price": 0.81,
                            "quantity": 1000,
                            "sizing_mode": "close",
                        },
                        ctx,
                    )
                    self.assertTrue(close_res["ok"], f"Close order failed: {close_res}")


class ProfileHorizonAndOrderNotionalTests(unittest.TestCase):
    def _base_env(self, td: str) -> dict:
        return {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            "FORESEA_AGENT_ACCOUNT_VALUE": "10000",
            "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.50",
            "FORESEA_AGENT_CLUSTER_CONCENTRATION_LIMIT": "0.50",
            "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_DAILY_RISK_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_MAX_OPEN_MARKETS": "20",
            "FORESEA_AGENT_MAX_TRADES_PER_CYCLE": "10",
            "FORESEA_MAX_ORDER_NOTIONAL": "2000.0",
        }

    def test_qwen_rejects_under_7d_short_horizon_news(self):
        ctx = benchmark_tools.ToolContext(agent_id="qwen3-8-27b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXFASTNEWS", bid=0.20, ask=0.22, prob=0.21, lead_days=4.5),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                result = place_trade(
                    {
                        "ticker": "KXFASTNEWS",
                        "side": "yes",
                        "price": 0.22,
                        "lead_days": 4.5,
                        "sizing_mode": "scaled_edge",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "profile_horizon_restricted")
                self.assertIn("profile_horizon_restricted", result["message"])

    def test_qwen_allows_14_to_30d_horizon(self):
        ctx = benchmark_tools.ToolContext(agent_id="qwen3-8-27b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXMACRO", bid=0.20, ask=0.22, prob=0.21, lead_days=18.0),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                result = place_trade(
                    {
                        "ticker": "KXMACRO",
                        "side": "yes",
                        "price": 0.22,
                        "lead_days": 18.0,
                        "sizing_mode": "scaled_edge",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(result["ok"], f"Trade failed: {result}")

    def test_short_horizon_weather_is_exempt(self):
        ctx = benchmark_tools.ToolContext(agent_id="qwen3-8-27b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote(
                        "KXWEATHER",
                        bid=0.20,
                        ask=0.22,
                        prob=0.21,
                        lead_days=2.0,
                        category="weather",
                        resolution_criteria="The Weather Company reports station KORD.",
                    ),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                result = place_trade(
                    {
                        "ticker": "KXWEATHER",
                        "side": "yes",
                        "price": 0.22,
                        "lead_days": 2.0,
                        "category": "weather",
                        "sizing_mode": "scaled_edge",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(result["ok"], f"Weather trade failed: {result}")

    def test_deepseek_rejects_single_order_exceeding_2pct_notional_cap(self):
        ctx = benchmark_tools.ToolContext(agent_id="deepseek-v4-flash", require_kelly_sizing=False)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXPROBE", bid=0.20, ask=0.25, prob=0.22, lead_days=14.0),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                # 2% of $10,000 is $200. Sizing 1200 contracts @ 0.25 = $300 > $200 cap.
                result = place_trade(
                    {
                        "ticker": "KXPROBE",
                        "side": "yes",
                        "price": 0.25,
                        "quantity": 1200,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "profile_order_notional_exceeded")
                self.assertIn("profile_order_notional_exceeded", result["message"])

    def test_deepseek_allows_single_order_under_2pct_notional_cap(self):
        ctx = benchmark_tools.ToolContext(agent_id="deepseek-v4-flash", require_kelly_sizing=False)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXPROBE", bid=0.20, ask=0.25, prob=0.22, lead_days=14.0),
                ),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi_orderbook",
                    return_value=_book(),
                ),
            ):
                # 700 contracts @ 0.25 = $175 <= $200 cap.
                result = place_trade(
                    {
                        "ticker": "KXPROBE",
                        "side": "yes",
                        "price": 0.25,
                        "quantity": 700,
                        "sizing_mode": "manual",
                        "model_probability": 0.35,
                    },
                    ctx,
                )
                self.assertTrue(result["ok"], f"Trade failed: {result}")


class AgentTickConfigAndSortingTests(unittest.TestCase):
    def test_configure_max_order_notional_scales_by_agent_profile(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_ACCOUNT_VALUE": "10000"}, clear=False):
            # Default no-agent call preserves 8%
            self.assertAlmostEqual(agent_trading_tick._configure_max_order_notional(), 800.0)

            # Specialized models scale to their profile:
            self.assertAlmostEqual(
                agent_trading_tick._configure_max_order_notional(agent_id="qwen3-8-27b"), 300.0
            )
            self.assertAlmostEqual(
                agent_trading_tick._configure_max_order_notional(agent_id="deepseek-v4-flash"), 200.0
            )
            self.assertAlmostEqual(
                agent_trading_tick._configure_max_order_notional(agent_id="minimax-m3"), 200.0
            )


class TradingTickQuantizationTests(unittest.TestCase):
    def test_polymarket_tick_quantization_tolerates_float_noise(self):
        # 0.8400000000000001 with tick_size 0.01 should cleanly quantize to 0.84
        order = trading.preview_order({
            "platform": "polymarket",
            "action": "buy",
            "token_id": "0x1234567890abcdef1234567890abcdef12345678",
            "price": "0.8400000000000001",
            "quantity": 10,
            "tick_size": "0.01",
            "time_in_force": "GTC",
        })
        self.assertAlmostEqual(order["normalized_order"]["price"], 0.84)

    def test_polymarket_tick_quantization_rejects_genuine_unaligned_price(self):
        # 0.845 with tick_size 0.01 should raise TradingValidationError
        with self.assertRaises(trading.TradingValidationError):
            trading.preview_order({
                "platform": "polymarket",
                "action": "buy",
                "token_id": "0x1234567890abcdef1234567890abcdef12345678",
                "price": "0.845",
                "quantity": 10,
                "tick_size": "0.01",
                "time_in_force": "GTC",
            })


if __name__ == "__main__":
    unittest.main()
