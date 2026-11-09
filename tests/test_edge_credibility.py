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

    def test_past_due_market_rejection(self):
        opp = {
            "question": "Will candidate win debate on 2026-09-01?",
            "platform": "Polymarket",
            "market_probability": 0.40,
            "model_probability": 0.60,
            "resolve_time": "2026-09-01T00:00:00Z",
            "evidence": [{"title": "Debate concluded"}],
            "volume": 50000,
        }
        res = audit_edge_opportunity(opp)
        self.assertIn("potential_past_due_market", res["credibility_flags"])
        self.assertLessEqual(res["credibility_score"], 0.60)

    def test_wide_bid_ask_spread_penalty(self):
        opp = {
            "question": "Thin market with wide spread",
            "platform": "Kalshi",
            "market_probability": 0.50,
            "model_probability": 0.70,
            "market_bid": 0.15,
            "market_ask": 0.85,
            "volume": 300,
        }
        res = audit_edge_opportunity(opp)
        self.assertIn("wide_bid_ask_spread", res["credibility_flags"])
        self.assertLess(res["credibility_score"], 0.70)


def _board_row(**overrides):
    # The artwork row the live board graded A on 2026-11-09: model 5%, market 93%.
    row = {
        "question": "Will an artwork sell for $150 million by December 31?",
        "model_probability": 0.05, "market_probability": 0.93, "market_bid": 0.90, "market_ask": 0.95,
        "market_volume": 69.3, "market_liquidity": 1889.7, "evidence_count": 5,
        "resolution_criteria": "Resolves YES if any single lot sells at a major auction house for at least $150M.",
        "resolve_time": "2099-01-01T04:59:00Z", "horizon": "30d+",
        "discrepancy_status": "extreme_forecast", "review_reasons": ["extreme_model_probability"],
        "needs_discrepancy_review": True,
    }
    row.update(overrides)
    return row


class BoardReviewTests(unittest.TestCase):
    def test_a_row_the_board_flags_for_review_is_never_graded_credible(self):
        res = audit_edge_opportunity(_board_row())
        self.assertEqual(res["credibility_grade"], "C")
        self.assertFalse(res["is_credible"])
        self.assertLess(res["credibility_score"], 0.60)
        self.assertIn("board_review_extreme_forecast", res["credibility_flags"])
        self.assertIn("extreme model probability", res["audit_summary"])
        self.assertNotIn("high_discrepancy_well_evidenced", res["credibility_flags"])

    def test_every_review_status_caps_the_grade(self):
        for status in ("extreme_forecast", "stale_forecast", "context_incomplete", "thin_market", "wide_spread"):
            res = audit_edge_opportunity(_board_row(model_probability=0.60, market_probability=0.40,
                                                    discrepancy_status=status, review_reasons=[]))
            self.assertEqual(res["credibility_grade"], "C", status)

    def test_a_genuine_candidate_keeps_its_grade(self):
        res = audit_edge_opportunity(_board_row(model_probability=0.60, market_probability=0.40, market_volume=8000,
                                                discrepancy_status="genuine_candidate", review_reasons=[]))
        self.assertEqual(res["credibility_grade"], "A")

    def test_an_extreme_gap_costs_score_even_with_evidence(self):
        row = _board_row(discrepancy_status=None, market_volume=None, resolution_criteria="", question="Will X happen?")
        res = audit_edge_opportunity(row)
        calm = audit_edge_opportunity({**row, "model_probability": 0.80})
        self.assertIn("extreme_edge_unverified", res["credibility_flags"])
        self.assertAlmostEqual(calm["credibility_score"] - res["credibility_score"], 0.15)


if __name__ == "__main__":
    unittest.main()
