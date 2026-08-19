"""Unit tests for Foresea Adversarial Debate Engine."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.debate_engine import (  # noqa: E402
    AdversarialDebateEngine,
    conduct_market_debate,
)


class DebateEngineTests(unittest.TestCase):
    def test_conduct_market_debate_analytical(self):
        res = conduct_market_debate(
            question="Will the Federal Reserve cut interest rates in September 2026?",
            platform="Kalshi",
            market_prob=0.35,
            evidence=[{"title": "Inflation data cools down to 2.4% annualized"}],
            resolution_criteria="Resolves YES if FOMC lowers target range.",
        )

        self.assertIn("bull_agent", res)
        self.assertIn("bear_agent", res)
        self.assertIn("chief_risk_judge", res)
        self.assertIn("synthesized_probability", res)
        self.assertIn("recommendation", res)
        self.assertEqual(res["bull_agent"]["stance"], "YES")
        self.assertEqual(res["bear_agent"]["stance"], "NO")
        self.assertGreater(len(res["chief_risk_judge"]["blind_spots"]), 0)
        self.assertAlmostEqual(res["market_probability"], 0.35)

    def test_debate_with_no_evidence(self):
        engine = AdversarialDebateEngine()
        res = engine.execute_debate(
            question="Will SpaceX reach Mars orbit before 2028?",
            platform="Polymarket",
            market_prob=0.20,
        )
        self.assertIsInstance(res["synthesized_probability"], float)
        self.assertIn(res["recommendation"], ["BUY YES", "BUY NO", "HOLD / NEUTRAL"])


if __name__ == "__main__":
    unittest.main()
