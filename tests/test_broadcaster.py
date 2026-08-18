"""Unit tests for Foresea Multi-Channel Broadcaster."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from channel_broadcaster import ChannelBroadcaster  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int = 200, data: Any = None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class ChannelBroadcasterTests(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.broadcaster = ChannelBroadcaster(
            foresea_url="https://foresea.test",
            telegram_token="test_tg_token",
            config_path=None,
            dry_run=True,
            session=self.mock_session,
        )

    def test_format_telegram_alert(self):
        opps = [
            {
                "question": "Will BTC reach $100k?",
                "platform": "Polymarket",
                "market_probability": 0.40,
                "model_probability": 0.65,
                "edge": 0.25,
                "recommendation": "BUY YES",
            }
        ]
        msg = self.broadcaster.format_telegram_alert(opps)
        self.assertIn("FORESEA MARKET EDGE ALERT", msg)
        self.assertIn("Will BTC reach $100k?", msg)
        self.assertIn("Polymarket", msg)
        self.assertIn("+25.0% edge", msg)

    def test_format_discord_embed(self):
        opps = [
            {
                "question": "Will Fed cut 50bps?",
                "platform": "Kalshi",
                "market_probability": 0.15,
                "model_probability": 0.35,
                "edge": 0.20,
                "recommendation": "BUY YES",
            }
        ]
        embed = self.broadcaster.format_discord_embed(opps)
        self.assertEqual(embed["title"], "⚡ Foresea Market Edge Alert")
        self.assertEqual(len(embed["fields"]), 1)
        self.assertIn("Kalshi", embed["fields"][0]["name"])

    def test_run_broadcast_dry_run_simulation(self):
        self.mock_session.get.return_value = MockResponse(
            200,
            {
                "opportunities": [
                    {
                        "question": "Test Market",
                        "market_probability": 0.20,
                        "model_probability": 0.35,
                    }
                ]
            },
        )
        self.broadcaster.load_targets = MagicMock(
            return_value={
                "discord_webhooks": [{"url": "https://discord.test/hook", "name": "Test Hook"}],
                "telegram_channels": [{"chat_id": "@TestChannel"}],
            }
        )
        stats = self.broadcaster.run_broadcast(min_edge=0.05)
        self.assertEqual(stats["sent"], 2)
        self.assertEqual(stats["failed"], 0)


if __name__ == "__main__":
    unittest.main()
