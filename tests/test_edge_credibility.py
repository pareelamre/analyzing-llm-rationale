"""Unit tests for Foresea Edge Credibility and Verification Engine."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.edge_credibility import (  # noqa: E402
    audit_edge_board,
    audit_edge_opportunity,
)


class EdgeCredibilityTests(unittest.TestCase):
    def test_high_credibility_opportunity(self):
        opp = {
            "question": "Will SpaceX launch Starship Flight 6 by December 2026?",
            "platform": "Polymarket",
            "market_probability": 0.40,
            "model_probability": 0.58,  # +18% edge
            "resolution_criteria": "Resolves YES if SpaceX conducts Starship orbital launch test before Dec 31, 2026 according to FAA and SpaceX official announcements.",
            "evidence": [{"title": "FAA issues launch license for Flight 6"}],
            "volume": 25000,
        }
        res = audit_edge_opportunity(opp)
        self.assertGreaterEqual(res["credibility_score"], 0.80)
        self.assertEqual(res["credibility_grade"], "A")
        self.assertTrue(res["is_credible"])
        self.assertIn("healthy_market_liquidity", res["credibility_flags"])

    def test_unverified_extreme_edge_penalty(self):
        opp = {
            "question": "Will obscure event occur?",
            "platform": "Kalshi",
            "market_probability": 0.05,
            "model_probability": 0.95,  # 90% edge with 0 evidence
            "evidence": [],
            "resolution_criteria": "",
        }
        res = audit_edge_opportunity(opp)
        self.assertLess(res["credibility_score"], 0.70)
        self.assertIn("extreme_edge_sparse_evidence", res["credibility_flags"])

    def test_missing_data_rejection(self):
        opp = {
            "question": "Incomplete market",
            "market_probability": None,
            "model_probability": 0.50,
        }
        res = audit_edge_opportunity(opp)
        self.assertEqual(res["credibility_score"], 0.0)
        self.assertEqual(res["credibility_grade"], "C")
        self.assertFalse(res["is_credible"])

    def test_audit_edge_board_batch(self):
        opps = [
            {"question": "Q1", "market_probability": 0.30, "model_probability": 0.45},
            {"question": "Q2", "market_probability": 0.50, "model_probability": 0.52},
        ]
        audited = audit_edge_board(opps)
        self.assertEqual(len(audited), 2)
        self.assertIn("credibility_score", audited[0])
        self.assertIn("credibility_grade", audited[1])


if __name__ == "__main__":
    unittest.main()
