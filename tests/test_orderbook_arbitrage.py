"""Tests for depth-aware binary complement-arbitrage calculations."""
from __future__ import annotations

import unittest

from analyzing_llm_rationale.orderbook_arbitrage import (
    kalshi_complement_ask_levels,
    polymarket_ask_levels,
    scan_complement_arbitrage,
)


class OrderbookArbitrageTests(unittest.TestCase):
    def test_polymarket_uses_ask_depth_and_matches_available_quantity(self):
        yes_asks = polymarket_ask_levels({"asks": [{"price": "0.42", "size": "3"}]})
        no_asks = polymarket_ask_levels({"asks": [{"price": "0.51", "size": "2"}]})

        result = scan_complement_arbitrage(yes_asks, no_asks, fee_bps_per_leg=10)

        self.assertTrue(result["candidate"])
        self.assertEqual(result["executable_quantity"], 2.0)
        self.assertAlmostEqual(result["levels"][0]["entry_cost"], 0.93)
        self.assertAlmostEqual(result["levels"][0]["fees_per_pair"], 0.00093)
        self.assertAlmostEqual(result["levels"][0]["net_edge_per_pair"], 0.06907)

    def test_kalshi_converts_complementary_bids_to_tradeable_asks(self):
        yes_asks, no_asks = kalshi_complement_ask_levels({
            "yes": [[0.47, 4]],
            "no": [[0.58, 5]],
        })

        self.assertEqual(yes_asks, [(0.42000000000000004, 5.0)])
        self.assertEqual(no_asks, [(0.53, 4.0)])
        result = scan_complement_arbitrage(yes_asks, no_asks, min_net_edge=0.04)
        self.assertTrue(result["candidate"])
        self.assertEqual(result["executable_quantity"], 4.0)
        self.assertAlmostEqual(result["estimated_net_profit"], 0.2)

    def test_fees_can_eliminate_an_apparent_spread(self):
        result = scan_complement_arbitrage([(0.49, 10)], [(0.50, 10)], fee_bps_per_leg=150)

        self.assertFalse(result["candidate"])
        self.assertEqual(result["executable_quantity"], 0)
