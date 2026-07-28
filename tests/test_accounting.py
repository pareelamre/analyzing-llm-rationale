from __future__ import annotations

import unittest
from datetime import datetime, timezone

from analyzing_llm_rationale.accounting import (
    NO,
    YES,
    PredictionMarketAccount,
    simulate_mark_to_market_account,
)


class PredictionMarketAccountTests(unittest.TestCase):
    def test_open_positions_mark_to_bid_liquidation_value(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(
            platform="Kalshi",
            ident="KXTEST",
            side=YES,
            quantity=10,
            price=0.60,
            ts=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )

        snap = account.snapshot({
            ("Kalshi", "KXTEST"): {"market_bid": 0.48, "market_ask": 0.52},
        })

        self.assertEqual(snap["value_method"], "mark_to_market_bid_liquidation")
        self.assertEqual(snap["cash"], 94.0)
        self.assertEqual(snap["liquidation_value"], 4.8)
        self.assertEqual(snap["account_value"], 98.8)
        self.assertEqual(snap["unrealized_pnl"], -1.2)

    def test_buying_opposite_side_realizes_kalshi_pair_netting(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(platform="Kalshi", ident="KXTEST", side=YES, quantity=10, price=0.60)

        fill = account.buy(
            platform="Kalshi",
            ident="KXTEST",
            side=NO,
            quantity=10,
            price=0.35,
            fee=0.17,
        )

        self.assertEqual(fill.settlement_status, "realized")
        self.assertEqual(fill.realized_pairs, 10)
        self.assertAlmostEqual(fill.realized_pnl, 0.33)
        self.assertEqual(account.open_positions(), [])
        self.assertAlmostEqual(account.cash, 100.33)

    def test_simulation_tracks_value_curve(self):
        rows = [
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "model_probability": 0.7,
                "market_probability": 0.6,
                "market_bid": 0.58,
                "market_ask": 0.62,
                "snapshot_ts": "2026-07-01T12:00:00+00:00",
            }
        ]

        account = simulate_mark_to_market_account(
            rows,
            latest_quotes={("Kalshi", "KXTEST"): {"market_bid": 0.50, "market_ask": 0.56}},
        )

        self.assertEqual(account["n_trades"], 1)
        self.assertEqual(account["liquidation_value"], 0.5)
        self.assertEqual(account["account_value"], 9999.88)
        self.assertEqual(len(account["value_curve"]), 1)


if __name__ == "__main__":
    unittest.main()
