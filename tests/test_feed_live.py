"""End-to-end test verifying the Foresea Alpha & Agent Feed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from channel_broadcaster import ChannelBroadcaster  # noqa: E402
from foresea_discord_bot import ForeseaDiscordClient  # noqa: E402
from foresea_telegram_bot import ForeseaTelegramBot  # noqa: E402

from analyzing_llm_rationale.server import app  # noqa: E402


class FeedEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.broadcaster = ChannelBroadcaster(dry_run=True)
        self.discord_client = ForeseaDiscordClient(session=self.client)
        self.tg_bot = ForeseaTelegramBot(bot_token="mock_token", session=self.client)

    def test_feed_endpoint_schema_and_content(self):
        resp = self.client.get("/feed/latest?limit=5&min_edge=0.01")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        
        self.assertIn("timestamp", data)
        self.assertIn("channels", data)
        self.assertIn("market_edge_signals", data)
        self.assertIn("agent_trades", data)
        self.assertIn("leaderboard_summary", data)

        discord_info = data["channels"]["discord"]
        self.assertEqual(discord_info["guild_id"], "1539674155228860527")
        self.assertEqual(discord_info["channel_id"], "1539674155799289991")
        self.assertEqual(data["channels"]["telegram"]["invite_url"], "https://t.me/+QIVxIyqCc-w4NzQ9")

    def test_discord_feed_embed_generation(self):
        embed = self.discord_client.build_feed_embed()
        self.assertEqual(embed["title"], "🌊 Foresea Alpha & Agent Live Feed")
        self.assertIn("fields", embed)
        self.assertTrue(len(embed["fields"]) >= 1)

    def test_telegram_feed_message_generation(self):
        msg = self.tg_bot.handle_feed()
        self.assertIn("Foresea Alpha & Agent Live Feed", msg)
        self.assertIn("Top Edge Signals", msg)


if __name__ == "__main__":
    unittest.main()
