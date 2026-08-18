"""Unit tests for Foresea Quant Execution Bridge."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from live_trader_bridge import LiveTraderBridge, RiskLimits  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int = 200, data: Any = None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class LiveTraderBridgeTests(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.risk = RiskLimits(
            max_position_usd=50.0,
            min_edge_threshold=0.08,
            max_total_allocation_usd=200.0,
        )
        self.bridge = LiveTraderBridge(
            risk=self.risk,
            foresea_url="https://foresea.test",
            dry_run=True,
            session=self.mock_session,
        )

    def test_evaluate_opportunity_filters_low_edge(self):
        opp = {
            "platform": "Kalshi",
            "ticker": "KXFED",
            "market_probability": 0.50,
            "model_probability": 0.52,  # edge is only 2%, below 8%
        }
        res = self.bridge.evaluate_opportunity(opp)
        self.assertIsNone(res)

    def test_evaluate_opportunity_generates_order_intent(self):
        opp = {
            "platform": "Kalshi",
            "ticker": "KXFED-25MAY",
            "market_probability": 0.30,
            "model_probability": 0.55,  # edge is 25% > 8%
            "question": "Will Fed cut rates?",
        }
        res = self.bridge.evaluate_opportunity(opp)
        self.assertIsNotNone(res)
        self.assertEqual(res["platform"], "kalshi")
        self.assertEqual(res["side"], "yes")
        self.assertEqual(res["target_price"], 0.30)
        self.assertGreater(res["contracts"], 0)
        self.assertLessEqual(res["cost_usd"], 50.0)

    def test_execute_order_dry_run_records_allocation(self):
        intent = {
            "platform": "kalshi",
            "ticker": "KXFED-25MAY",
            "side": "yes",
            "target_price": 0.40,
            "contracts": 100,
            "cost_usd": 40.0,
            "edge": 0.20,
            "question": "Test question",
        }
        result = self.bridge.execute_order(intent)
        self.assertEqual(result["status"], "simulated")
        self.assertEqual(self.risk.allocated_usd, 40.0)

    def test_risk_limits_prevent_overallocation(self):
        self.risk.allocated_usd = 190.0  # cap is 200.0
        self.assertFalse(self.risk.can_allocate(25.0))
        self.assertTrue(self.risk.can_allocate(10.0))


if __name__ == "__main__":
    unittest.main()
