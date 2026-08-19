"""Unit tests for Foresea Whale Flow & Smart Money Intelligence."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server import app  # noqa: E402
from analyzing_llm_rationale.whale_flow import analyze_whale_trades  # noqa: E402


class WhaleFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_analyze_whale_trades_sentiment(self):
        trades = [
            {"platform": "Polymarket", "price": 0.40, "size": 2500, "side": "YES", "question": "Fed rate cut"},
            {"platform": "Kalshi", "price": 0.20, "size": 5000, "side": "YES", "question": "CPI < 2.5%"},
            {"platform": "Polymarket", "price": 0.50, "size": 600, "side": "NO", "question": "Election winner"},
        ]
        res = analyze_whale_trades(trades, min_notional_usd=250.0)
        self.assertEqual(res["n_whale_prints"], 3)
        self.assertGreater(res["total_whale_volume_usd"], 2000.0)
        self.assertEqual(res["whale_sentiment_label"], "BULLISH ACCUMULATION")
        self.assertGreaterEqual(res["whale_sentiment_index_pct"], 65.0)

    def test_market_whale_flow_endpoint(self):
        with patch("analyzing_llm_rationale.whale_flow.fetch_live_whale_flow") as mock_flow:
            mock_flow.return_value = {
                "status": "ok",
                "total_whale_volume_usd": 15000.0,
                "whale_sentiment_label": "BULLISH ACCUMULATION",
                "top_prints": [],
            }
            resp = self.client.get("/v1/market/whale-flow?min_notional=500")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["total_whale_volume_usd"], 15000.0)


if __name__ == "__main__":
    unittest.main()
