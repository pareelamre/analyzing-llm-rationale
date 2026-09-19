"""Unit tests for Foresea agent recalibration and specialized market profiles."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts import agent_trading_tick  # noqa: E402

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


def _quote(ident: str, bid: float = 0.20, ask: float = 0.22, platform: str = "kalshi",
           prob: float = 0.21, close_days: int = 20) -> dict:
    return {
        "platform": platform,
        "ident": ident,
        "ticker": ident,
        "question": f"Test market {ident}",
        "yes_bid": bid,
        "yes_ask": ask,
        "no_bid": round(1.0 - ask, 2),
        "no_ask": round(1.0 - bid, 2),
        "probability": prob,
        "close_days": close_days,
        "close_date": "2026-10-10",
        "open_date": "2026-09-01",
        "category": "politics",
    }


class AgentRecalibrationProfilesTests(unittest.TestCase):
    def test_all_fleet_models_have_specialization_profiles(self):
        fleet_models = [
            "qwen3-8-27b",
            "gemma-4-26b-a4b-it",
            "glm-5-3",
            "glm-5-3-flash",
            "deepseek-v4-flash",
            "minimax-m3",
            "gpt-oss-120b",
            "llama-3.3-70b-instruct",
        ]
        for model in fleet_models:
            profile = benchmark_tools.get_agent_profile(model)
            self.assertIsNotNone(profile, f"Missing profile for model {model}")
            self.assertTrue(profile.role_title)
            self.assertTrue(profile.tactical_mandate)

    def test_profile_resolution_handles_provider_prefixes_and_casing(self):
        self.assertEqual(
            benchmark_tools.get_agent_profile("google/gemma-4-26B-A4B-it").model_id,
            "gemma-4-26b-a4b-it",
        )
        self.assertEqual(
            benchmark_tools.get_agent_profile("meta-llama/Llama-3.3-70B-Instruct").model_id,
            "llama-3.3-70b-instruct",
        )
        self.assertEqual(
            benchmark_tools.get_agent_profile("openai/gpt-oss-120b").model_id,
            "gpt-oss-120b",
        )
        self.assertEqual(
            benchmark_tools.get_agent_profile("THUDM/glm-5-3").model_id,
            "glm-5-3",
        )
        self.assertEqual(
            benchmark_tools.get_agent_profile("deepseek-ai/deepseek-v4-flash").model_id,
            "deepseek-v4-flash",
        )
        self.assertEqual(
            benchmark_tools.get_agent_profile("MiniMax/MiniMax-M3").model_id,
            "minimax-m3",
        )
        self.assertIsNone(benchmark_tools.get_agent_profile("unknown-custom-model"))

    def test_profile_calibrations_match_empirical_findings(self):
        gemma = benchmark_tools.get_agent_profile("gemma-4-26b-a4b-it")
        self.assertEqual(gemma.max_contract_price, 0.45)
        self.assertEqual(gemma.preferred_sizing_mode, "convex_conviction")
        self.assertEqual(gemma.horizon_preference, "underpriced_skew")

        deepseek = benchmark_tools.get_agent_profile("deepseek-v4-flash")
        self.assertEqual(deepseek.max_contract_price, 0.70)
        self.assertEqual(deepseek.preferred_sizing_mode, "probe_kelly")

        minimax = benchmark_tools.get_agent_profile("minimax-m3")
        self.assertEqual(minimax.forbidden_price_range, (0.50, 0.75))
        self.assertEqual(minimax.preferred_sizing_mode, "flat_probe")

        gpt = benchmark_tools.get_agent_profile("gpt-oss-120b")
        self.assertEqual(gpt.max_trades_per_cycle, 1)
        self.assertEqual(gpt.min_profile_edge, 0.05)
        self.assertEqual(gpt.preferred_sizing_mode, "edge_kelly")

        llama = benchmark_tools.get_agent_profile("llama-3.3-70b-instruct")
        self.assertEqual(llama.max_contract_price, 0.40)
        self.assertEqual(llama.horizon_preference, "14-30d")
        self.assertEqual(llama.preferred_sizing_mode, "convex_conviction")

        qwen = benchmark_tools.get_agent_profile("qwen3-8-27b")
        self.assertEqual(qwen.horizon_preference, "14-30d")
        self.assertEqual(qwen.preferred_sizing_mode, "convex_conviction")

        glm = benchmark_tools.get_agent_profile("glm-5-3")
        self.assertEqual(glm.min_profile_edge, 0.04)
        self.assertEqual(glm.horizon_preference, "14-30d")


class AgentSpecializationRiskGuardsTests(unittest.TestCase):
    def _base_env(self, td: str) -> dict:
        return {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(td) / "ledger.jsonl"),
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(td) / "accounts.sqlite"),
            "FORESEA_AGENT_ACCOUNT_VALUE": "1000",
            "FORESEA_AGENT_CONCENTRATION_LIMIT": "0.50",
            "FORESEA_AGENT_PER_CYCLE_SPEND_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_DAILY_RISK_LIMIT_PCT": "0.50",
            "FORESEA_AGENT_MAX_TRADES_PER_CYCLE": "5",
            "FORESEA_MAX_ORDER_NOTIONAL": "200.0",
            "FORESEA_AGENT_CYCLE_ID": "cycle-recal-1",
        }

    def test_gemma_rejects_price_exceeding_positive_skew_ceiling(self):
        ctx = benchmark_tools.ToolContext(agent_id="gemma-4-26b-a4b-it", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXEXPENSIVE", bid=0.50, ask=0.52, prob=0.51),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXEXPENSIVE", "side": "yes", "price": 0.52,
                     "sizing_mode": "convex_conviction", "model_probability": 0.70},
                    ctx,
                )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "profile_price_ceiling_exceeded")
        self.assertIn("profile_price_ceiling_exceeded", result["message"])

    def test_gemma_allows_price_below_ceiling(self):
        ctx = benchmark_tools.ToolContext(agent_id="gemma-4-26b-a4b-it", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXCHEAP", bid=0.20, ask=0.22, prob=0.21),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXCHEAP", "side": "yes", "price": 0.22,
                     "sizing_mode": "convex_conviction", "model_probability": 0.35},
                    ctx,
                )
        self.assertTrue(result["ok"])

    def test_deepseek_rejects_steamroller_risk_above_70c(self):
        ctx = benchmark_tools.ToolContext(agent_id="deepseek-v4-flash", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXSTEAM", bid=0.78, ask=0.80, prob=0.79),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXSTEAM", "side": "yes", "price": 0.80,
                     "sizing_mode": "probe_kelly", "model_probability": 0.95},
                    ctx,
                )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "profile_price_ceiling_exceeded")

    def test_minimax_rejects_forbidden_50_to_75c_coin_flip_band(self):
        ctx = benchmark_tools.ToolContext(agent_id="minimax-m3", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXCOINFLIP", bid=0.58, ask=0.60, prob=0.59),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXCOINFLIP", "side": "yes", "price": 0.60,
                     "sizing_mode": "flat_probe", "model_probability": 0.75},
                    ctx,
                )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "profile_price_band_forbidden")
        self.assertIn("profile_price_band_forbidden", result["message"])

    def test_minimax_allows_outside_forbidden_band(self):
        ctx = benchmark_tools.ToolContext(agent_id="minimax-m3", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXASYMM", bid=0.25, ask=0.27, prob=0.26),
                ),
            ):
                result = benchmark_tools.place_trade(
                    {"ticker": "KXASYMM", "side": "yes", "price": 0.27,
                     "sizing_mode": "flat_probe", "model_probability": 0.35},
                    ctx,
                )
        self.assertTrue(result["ok"])

    def test_gpt_oss_rejects_thin_edge_below_5pp_hurdle(self):
        ctx = benchmark_tools.ToolContext(agent_id="gpt-oss-120b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXTHIN", bid=0.30, ask=0.32, prob=0.31),
                ),
            ):
                # Ask is 0.32, model prob is 0.35 -> net edge ~0.03 (3pp), below 5pp profile hurdle
                result = benchmark_tools.place_trade(
                    {"ticker": "KXTHIN", "side": "yes", "price": 0.32,
                     "sizing_mode": "scaled_edge", "model_probability": 0.35},
                    ctx,
                )
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "insufficient_profile_edge")
        self.assertIn("insufficient_profile_edge", result["message"])

    def test_gpt_oss_allows_edge_clearing_5pp_hurdle(self):
        ctx = benchmark_tools.ToolContext(agent_id="gpt-oss-120b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    return_value=_quote("KXFAT", bid=0.30, ask=0.32, prob=0.31),
                ),
            ):
                # Ask is 0.32, model prob is 0.40 -> net edge ~0.08 (8pp), clears 5pp hurdle
                result = benchmark_tools.place_trade(
                    {"ticker": "KXFAT", "side": "yes", "price": 0.32,
                     "sizing_mode": "scaled_edge", "model_probability": 0.40},
                    ctx,
                )
        self.assertTrue(result["ok"])

    def test_gpt_oss_enforces_one_trade_per_cycle_limit(self):
        ctx = benchmark_tools.ToolContext(agent_id="gpt-oss-120b", require_kelly_sizing=True)
        with tempfile.TemporaryDirectory() as td:
            with (
                mock.patch.dict(os.environ, self._base_env(td), clear=False),
                mock.patch(
                    "analyzing_llm_rationale.market_data.fetch_kalshi",
                    side_effect=lambda t: _quote(t, bid=0.20, ask=0.22, prob=0.21),
                ),
            ):
                first = benchmark_tools.place_trade(
                    {"ticker": "KXT1", "side": "yes", "price": 0.22,
                     "sizing_mode": "scaled_edge", "model_probability": 0.40},
                    ctx,
                )
                second = benchmark_tools.place_trade(
                    {"ticker": "KXT2", "side": "yes", "price": 0.22,
                     "sizing_mode": "scaled_edge", "model_probability": 0.40},
                    ctx,
                )
        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "trade_rate_limit")


class AgentTradingTickPromptAndDiscoveryTests(unittest.TestCase):
    def test_tactical_profile_block_renders_for_known_models(self):
        gemma_block = agent_trading_tick._agent_tactical_profile_block("gemma-4-26b-a4b-it")
        self.assertIn("Positive-Skew Asymmetric Value Sniper", gemma_block)
        self.assertIn("Price ceiling: <= $0.45", gemma_block)

        gpt_block = agent_trading_tick._agent_tactical_profile_block("gpt-oss-120b")
        self.assertIn("Disciplined Low-Turnover Specialist", gpt_block)
        self.assertIn("Max trades per cycle: 1", gpt_block)
        self.assertIn("Minimum net edge hurdle: >= 5.0%", gpt_block)

        minimax_block = agent_trading_tick._agent_tactical_profile_block("minimax-m3")
        self.assertIn("Forbidden price band: $0.50-$0.75", minimax_block)

        unknown_block = agent_trading_tick._agent_tactical_profile_block("unknown-test")
        self.assertEqual(unknown_block, "")

    def test_assemble_question_injects_profile_block(self):
        question = agent_trading_tick._assemble_question(
            "=== Your portfolio ===\nNone",
            "=== Candidate markets ===\n- [kalshi] KX1",
            agent_id="gemma-4-26b-a4b-it",
        )
        self.assertIn("YOUR SPECIALIZED TRADING PROFILE & TACTICAL MANDATE", question)
        self.assertIn("Positive-Skew Asymmetric Value Sniper", question)
        self.assertIn(agent_trading_tick._TRADING_INSTRUCTION, question)

    def test_candidate_discovery_prioritizes_gemma_underpriced_skew(self):
        cheap_quote = _quote("KXCHEAP", bid=0.18, ask=0.20, prob=0.19, close_days=5)
        mid_quote = _quote("KXMID", bid=0.58, ask=0.60, prob=0.59, close_days=20)
        with (
            mock.patch.object(agent_trading_tick, "_discover_weather_candidates", return_value=[]),
            mock.patch.object(agent_trading_tick, "_discover_mtm_edge_candidates", return_value=[]),
            mock.patch.object(agent_trading_tick, "_list_venue", side_effect=lambda p, lim: [cheap_quote, mid_quote] if p == "kalshi" else []),
        ):
            candidates = agent_trading_tick._discover_candidates(set(), agent_id="gemma-4-26b-a4b-it")
            self.assertTrue(len(candidates) >= 1)
            self.assertEqual(candidates[0]["ident"], "KXCHEAP")


if __name__ == "__main__":
    unittest.main()
