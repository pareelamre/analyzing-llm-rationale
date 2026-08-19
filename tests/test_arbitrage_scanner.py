"""Unit tests for Foresea Cross-Venue Arbitrage & Spread Scanner."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.arbitrage_scanner import (  # noqa: E402
    _compute_keyword_overlap,
    scan_cross_venue_arbitrage,
)


class ArbitrageScannerTests(unittest.TestCase):
    def test_normalize_and_overlap(self):
        t1 = "Will the Federal Reserve cut interest rates in September?"
        t2 = "Fed to cut interest rates at September FOMC meeting?"
        overlap = _compute_keyword_overlap(t1, t2)
        self.assertGreaterEqual(overlap, 0.40)

    def test_synthetic_arbitrage_detection(self):
        poly = [
            {
                "id": "poly-fed-sep",
                "question": "Will Federal Reserve cut rates in September?",
                "probability": 0.35,  # Cheap YES
                "market_url": "https://polymarket.com/market/1",
            }
        ]
        kalshi = [
            {
                "ticker": "KXFED-SEP",
                "question": "Federal Reserve to cut interest rates in September?",
                "probability": 0.50,  # Expensive YES -> Cheap NO (0.50)
                "market_url": "https://kalshi.com/market/2",
            }
        ]

        opps = scan_cross_venue_arbitrage(poly, kalshi, min_spread=0.05, min_overlap=0.30)
        self.assertEqual(len(opps), 1)
        opp = opps[0]
        self.assertAlmostEqual(opp["spread"], 0.15)
        self.assertEqual(opp["executable_action"]["long_venue"], "Polymarket")
        self.assertEqual(opp["executable_action"]["short_venue"], "Kalshi")
        self.assertGreater(opp["gross_roi_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
