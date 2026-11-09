"""Unit tests for Foresea V1 Enterprise API Endpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server import app  # noqa: E402


class V1ApiEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_v1_system_health(self):
        with patch("analyzing_llm_rationale.server._read_live_track_record", return_value={}), patch(
            "analyzing_llm_rationale.market_data.fetch_kalshi_exchange_status",
            return_value={"trading_active": True, "exchange_active": True},
        ), patch("analyzing_llm_rationale.market_data._get_json", return_value=[]):
            resp = self.client.get("/v1/system/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("venues", data)
        self.assertIn("polymarket", data["venues"])
        self.assertIn("kalshi", data["venues"])
        self.assertFalse(data["venues"]["polymarket"]["native_streaming"])
        self.assertIsNone(data["engine"]["brier_score_mean"])

    def test_system_health_reports_failed_venue_probe(self):
        from analyzing_llm_rationale.market_data import MarketDataError
        with patch("analyzing_llm_rationale.server._read_live_track_record", return_value={}), patch(
            "analyzing_llm_rationale.market_data.fetch_kalshi_exchange_status", side_effect=MarketDataError("offline"),
        ), patch("analyzing_llm_rationale.market_data._get_json", return_value=[]):
            response = self.client.get("/v1/system/health")
        self.assertEqual(response.json()["status"], "degraded")
        self.assertEqual(response.json()["venues"]["kalshi"]["status"], "unavailable")
        self.assertEqual(response.json()["venues"]["polymarket"]["status"], "connected")

    def test_v1_arbitrage_cross_venue(self):
        with patch("analyzing_llm_rationale.arbitrage_scanner.scan_cross_venue_arbitrage") as mock_arb:
            mock_arb.return_value = [
                {
                    "event_summary": "Fed Rate Cut",
                    "spread": 0.10,
                    "spread_pct": 10.0,
                    "strategy": "BUY YES on Polymarket + BUY NO on Kalshi",
                }
            ]
            resp = self.client.get("/v1/arbitrage/cross-venue?min_spread=0.03")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["n_opportunities"], 1)
            self.assertEqual(data["opportunities"][0]["spread"], 0.10)

    def test_v1_batch_forecast(self):
        with patch("analyzing_llm_rationale.server.predict") as mock_predict:
            from analyzing_llm_rationale.server import PredictResponse
            mock_predict.return_value = PredictResponse(
                question="Will GDP grow?",
                predicted_answer="YES",
                confidence=0.80,
                rationale="Solid expansion data.",
                variant="variant0_neutral_baseline",
                model_key="gpt-oss-120b",
            )
            payload = {
                "questions": [
                    {"id": "q1", "question": "Will GDP grow?", "market_probability": 0.50},
                    {"id": "q2", "question": "Will CPI drop?", "market_probability": 0.40},
                ]
            }
            resp = self.client.post("/v1/batch/forecast", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["n_total"], 2)
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["results"][0]["status"], "success")

    def test_v1_market_whale_flow(self):
        sample_flow = {
            "timestamp": "2026-11-08T20:00:00Z",
            "min_notional_usd": 250.0,
            "n_whale_prints": 1,
            "total_whale_volume_usd": 5000.0,
            "sentiment_index": 80.0,
            "sentiment_label": "BULLISH ACCUMULATION",
            "whale_prints": [{
                "platform": "Polymarket",
                "ticker": "will-fed-cut-rates",
                "market_title": "Will Fed Cut Rates?",
                "side": "YES",
                "price": 0.65,
                "size": 7692,
                "notional_usd": 5000.0,
                "timestamp": "2026-11-08T19:59:00Z",
            }],
        }
        with patch("analyzing_llm_rationale.whale_flow.fetch_live_whale_flow", return_value=sample_flow):
            resp = self.client.get("/v1/market/whale-flow?min_notional=250")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["n_whale_prints"], 1)
            self.assertEqual(data["sentiment_label"], "BULLISH ACCUMULATION")

    def test_openapi_tag_consistency(self):
        schema = app.openapi()
        tags = set()
        for _path, methods in schema.get("paths", {}).items():
            for _method, op in methods.items():
                if isinstance(op, dict):
                    for t in op.get("tags", []):
                        tags.add(t)
        self.assertNotIn("Agent", tags, "Use 'Agents' instead of singular 'Agent'")
        self.assertNotIn("Predict", tags, "Use 'Inference' instead of 'Predict'")


if __name__ == "__main__":
    unittest.main()
