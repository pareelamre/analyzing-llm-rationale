"""Unit tests for Foresea Telegram & Discord Bots."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from foresea_discord_bot import ForeseaDiscordClient  # noqa: E402
from foresea_telegram_bot import ForeseaTelegramBot  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int = 200, data: Any = None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class TelegramBotTests(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.bot = ForeseaTelegramBot(
            bot_token="test_token_123",
            foresea_base_url="https://foresea.test",
            session=self.mock_session,
        )

    def test_handle_forecast_formats_correctly(self):
        self.mock_session.post.return_value = MockResponse(
            200,
            {
                "predicted_probability": 0.78,
                "predicted_answer": "Yes",
                "rationale": "High likelihood based on recent data releases.",
                "evidence": [{"title": "Official Announcement in Q3"}],
            },
        )
        msg = self.bot.handle_forecast("Will X happen?")
        self.assertIn("Foresea Calibrated Forecast", msg)
        self.assertIn("78.0%", msg)
        self.assertIn("YES", msg)
        self.assertIn("Official Announcement in Q3", msg)

    def test_handle_edge_board_formats_opportunities(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "opportunities": [
                    {
                        "question": "Will Fed cut rates in May?",
                        "platform": "Polymarket",
                        "market_probability": 0.35,
                        "model_probability": 0.60,
                        "edge": 0.25,
                        "recommendation": "BUY YES",
                    }
                ]
            },
        )
        msg = self.bot.handle_edge_board()
        self.assertIn("Foresea Edge Board", msg)
        self.assertIn("Will Fed cut rates", msg)
        self.assertIn("Polymarket", msg)
        self.assertIn("+25.0% edge", msg)

    def test_handle_track_record_formats_metrics(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "n_snapshots_resolved": 42,
                "overall": {
                    "mean_brier_score": 0.1420,
                    "accuracy": 0.810,
                    "skill_vs_market": 0.035,
                },
            },
        )
        msg = self.bot.handle_track_record()
        self.assertIn("42", msg)
        self.assertIn("0.1420", msg)
        self.assertIn("81.0%", msg)

    def test_handle_feed(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "market_edge_signals": [
                    {
                        "question": "Will BTC reach $100k?",
                        "platform": "Polymarket",
                        "edge": 0.20,
                        "recommendation": "BUY YES",
                    }
                ],
                "agent_trades": [
                    {
                        "model": "gpt-oss-120b",
                        "action": "BUY",
                        "ticker": "BTC-100K",
                    }
                ],
            },
        )
        msg = self.bot.handle_feed()
        self.assertIn("Foresea Alpha & Agent Live Feed", msg)
        self.assertIn("BUY YES", msg)
        self.assertIn("gpt-oss-120b", msg)

    def test_handle_agents(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "leaderboard": [
                    {
                        "model": "qwen3-coder-30b",
                        "account_value": 11200.0,
                        "return_pct": 12.0,
                        "n_trades": 15,
                        "win_rate": 0.733,
                    }
                ]
            },
        )
        msg = self.bot.handle_agents()
        self.assertIn("Foresea Agent Trading Leaderboard", msg)
        self.assertIn("qwen3-coder-30b", msg)
        self.assertIn("$11,200.00", msg)

    def test_process_message_dispatch(self):
        self.mock_session.post.return_value = MockResponse(200, {"ok": True})
        self.bot.process_message({"chat": {"id": 999}, "text": "/subscribe"})
        self.assertIn(999, self.bot.subscribed_chats)


class DiscordBotTests(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.client = ForeseaDiscordClient(
            foresea_base_url="https://foresea.test",
            session=self.mock_session,
        )

    def test_build_forecast_embed(self):
        self.mock_session.post.return_value = MockResponse(
            200,
            {
                "predicted_probability": 0.42,
                "predicted_answer": "No",
                "rationale": "Model sees low probability.",
                "evidence": [{"title": "News Report"}],
            },
        )
        embed = self.client.build_forecast_embed("Will Team A win?")
        self.assertEqual(embed["title"], "🔮 Foresea Calibrated Forecast")
        self.assertIn("42.0%", embed["description"])
        self.assertIn("NO", embed["description"])
        self.assertEqual(len(embed["fields"]), 2)

    def test_build_edge_board_embed(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "opportunities": [
                    {
                        "question": "Will CPI exceed 3%?",
                        "platform": "Kalshi",
                        "market_probability": 0.50,
                        "model_probability": 0.70,
                        "recommendation": "BUY YES",
                    }
                ]
            },
        )
        embed = self.client.build_edge_board_embed()
        self.assertEqual(embed["title"], "⚡ Foresea Edge Board — Top Opportunities")
        self.assertEqual(len(embed["fields"]), 1)
        self.assertIn("Kalshi", embed["fields"][0]["name"])

    def test_build_feed_embed(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "market_edge_signals": [
                    {
                        "question": "Will CPI exceed 3%?",
                        "platform": "Kalshi",
                        "edge": 0.15,
                        "recommendation": "BUY YES",
                    }
                ],
                "agent_trades": [
                    {
                        "model": "kimi-k3",
                        "action": "BUY",
                        "ticker": "CPI-3PCT",
                        "shares": 100,
                    }
                ],
            },
        )
        embed = self.client.build_feed_embed()
        self.assertEqual(embed["title"], "🌊 Foresea Alpha & Agent Live Feed")
        self.assertEqual(len(embed["fields"]), 2)
        self.assertIn("Top Market Mispricings", embed["fields"][0]["name"])
        self.assertIn("Recent Agent Actions", embed["fields"][1]["name"])

    def test_build_agent_board_embed(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "leaderboard": [
                    {
                        "model": "gpt-oss-120b",
                        "account_value": 10500.0,
                        "return_pct": 5.0,
                        "n_trades": 8,
                        "win_rate": 0.625,
                    }
                ]
            },
        )
        embed = self.client.build_agent_board_embed()
        self.assertEqual(embed["title"], "🏆 Foresea Autonomous Agent Trading Leaderboard")
        self.assertEqual(len(embed["fields"]), 1)
        self.assertIn("gpt-oss-120b", embed["fields"][0]["name"])


if __name__ == "__main__":
    unittest.main()

