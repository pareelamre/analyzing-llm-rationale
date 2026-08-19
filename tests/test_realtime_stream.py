"""Unit tests for Foresea Debate, Portfolio, and Real-Time Stream endpoints."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.server import app  # noqa: E402


class RealtimeStreamAndEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_agent_debate_endpoint(self):
        resp = self.client.post(
            "/agent/debate",
            json={
                "question": "Will Ethereum reach $5000 before end of year?",
                "platform": "Polymarket",
                "market_probability": 0.25,
                "evidence": [{"title": "Layer 2 transaction volume hits new ATH"}],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("bull_agent", data)
        self.assertIn("bear_agent", data)
        self.assertIn("chief_risk_judge", data)
        self.assertIn("synthesized_probability", data)

    def test_portfolio_optimal_allocation_endpoint(self):
        with patch("analyzing_llm_rationale.server._read_edge_board_record") as mock_board:
            mock_board.return_value = {
                "edge_board": [
                    {
                        "question": "Will US GDP grow > 2.5%?",
                        "platform": "Kalshi",
                        "market_probability": 0.40,
                        "model_probability": 0.60,
                        "credibility_score": 0.90,
                    }
                ]
            }
            resp = self.client.get("/portfolio/optimal-allocation?bankroll=2000&kelly_fraction=0.25")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["bankroll_usd"], 2000.0)
            self.assertGreater(data["allocated_usd"], 0)

    def test_portfolio_optimize_post_endpoint(self):
        resp = self.client.post(
            "/portfolio/optimize",
            json={
                "bankroll_usd": 5000.0,
                "kelly_fraction": 0.50,
                "min_edge": 0.05,
                "opportunities": [
                    {
                        "question": "Candidate Alpha Market",
                        "platform": "Polymarket",
                        "market_probability": 0.30,
                        "model_probability": 0.55,
                        "credibility_score": 0.85,
                    }
                ],
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["bankroll_usd"], 5000.0)
        self.assertEqual(data["n_positions"], 1)

    def test_stream_radar_sse_registered(self):
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        self.assertIn("/stream/radar", route_paths)
        self.assertIn("/ws/radar", route_paths)


if __name__ == "__main__":
    unittest.main()
