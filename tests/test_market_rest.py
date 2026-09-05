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
        mock_fn.return_value = [{"price": 0.50, "count": 10}]
        resp = self.client.get("/market/trades", params={"platform": "kalshi", "ident": "KXTEST"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["trades"]), 1)

    @patch("analyzing_llm_rationale.market_data.fetch_trader_leaderboard")
    def test_leaderboard_route(self, mock_fn):
        mock_fn.return_value = [{"rank": 1, "username": "TopTrader"}]
        resp = self.client.get("/market/leaderboard")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["leaderboard"][0]["rank"], 1)

    @patch("analyzing_llm_rationale.market_data.fetch_polymarket_tags")
    def test_tags_route(self, mock_fn):
        mock_fn.return_value = [{"label": "finance"}, {"label": "fed"}]
        resp = self.client.get("/market/tags")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["tags"][0]["label"], "finance")

    def test_list_routes_preserve_envelopes_for_empty_and_nonempty_data(self):
        cases = [
            ("fetch_recent_trades", "/market/trades?ident=KXTEST", "trades"),
            ("fetch_trader_leaderboard", "/market/leaderboard", "leaderboard"),
            ("fetch_polymarket_tags", "/market/tags", "tags"),
            ("fetch_polymarket_price_history", "/market/price-history?platform=polymarket&ident=123", "candlesticks"),
        ]
        from analyzing_llm_rationale.market_data import MarketDataError
        for helper, url, key in cases:
            for rows in ([], [{"id": "sample"}]):
                with self.subTest(helper=helper, rows=rows), patch(
                    f"analyzing_llm_rationale.market_data.{helper}", return_value=rows,
                ):
                    response = self.client.get(url)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()[key], rows)
            with self.subTest(helper=helper, failure=True), patch(
                f"analyzing_llm_rationale.market_data.{helper}", side_effect=MarketDataError("upstream unavailable"),
            ):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["error"], "venue_data_unavailable")

    def test_kalshi_candle_route_supplies_documented_window(self):
        with patch("analyzing_llm_rationale.market_data._get_json", return_value={"candlesticks": [{"end_period_ts": 200}]} ) as get:
            response = self.client.get("/market/price-history?ident=KXBTC15M-TEST&start_ts=100&end_ts=200&period_interval=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_args.kwargs["params"], {"start_ts": 100, "end_ts": 200, "period_interval": 1})
        self.assertEqual(len(response.json()["candlesticks"]), 1)

    def test_invalid_platform_does_not_call_a_venue(self):
        with patch("analyzing_llm_rationale.market_data._get_json") as get:
            response = self.client.get("/market/trades?platform=unknown&ident=KXTEST")
        self.assertEqual(response.status_code, 400)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
