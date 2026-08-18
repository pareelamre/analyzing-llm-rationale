"""Unit tests for Foresea Market REST endpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server import app  # noqa: E402


class MarketRestTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("analyzing_llm_rationale.market_data.fetch_kalshi_exchange_status")
    @patch("analyzing_llm_rationale.market_data.fetch_kalshi_exchange_schedule")
    def test_exchange_status_route(self, mock_sched, mock_status):
        mock_status.return_value = {"trading_active": True, "exchange_active": True}
        mock_sched.return_value = {"standard_hours": []}
        resp = self.client.get("/market/exchange-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["status"]["trading_active"])

    @patch("analyzing_llm_rationale.market_data.fetch_kalshi_orderbook")
    def test_orderbook_route(self, mock_fn):
        mock_fn.return_value = {"orderbook": {"yes": [[45, 100]]}}
        resp = self.client.get("/market/orderbook", params={"platform": "kalshi", "ident": "KXTEST"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("orderbook", resp.json())

    @patch("analyzing_llm_rationale.market_data.fetch_recent_trades")
    def test_trades_route(self, mock_fn):
        mock_fn.return_value = {"trades": [{"price": 0.50, "count": 10}]}
        resp = self.client.get("/market/trades", params={"platform": "kalshi", "ident": "KXTEST"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["trades"]), 1)

    @patch("analyzing_llm_rationale.market_data.fetch_trader_leaderboard")
    def test_leaderboard_route(self, mock_fn):
        mock_fn.return_value = {"leaderboard": [{"rank": 1, "username": "TopTrader"}]}
        resp = self.client.get("/market/leaderboard")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["leaderboard"][0]["rank"], 1)

    @patch("analyzing_llm_rationale.market_data.fetch_polymarket_tags")
    def test_tags_route(self, mock_fn):
        mock_fn.return_value = {"tags": ["finance", "fed"]}
        resp = self.client.get("/market/tags")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("finance", resp.json()["tags"])


if __name__ == "__main__":
    unittest.main()
