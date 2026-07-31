from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.accounting import (
    NO,
    YES,
    PredictionMarketAccount,
    simulate_mark_to_market_account,
    simulate_market_follow_mark_to_market_account,
    simulate_shadow_mark_to_market_account,
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

    def test_settlement_records_realized_pnl(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(platform="Kalshi", ident="KXTEST", side=YES, quantity=2, price=0.40)

        settlement = account.settle_market(
            platform="Kalshi",
            ident="KXTEST",
            outcome=1,
            ts=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(settlement["settled_contracts"], 2)
        self.assertEqual(settlement["payout"], 2)
        self.assertAlmostEqual(settlement["realized_pnl"], 1.20)
        self.assertAlmostEqual(account.realized_pnl, 1.20)
        self.assertAlmostEqual(account.cash, 101.20)

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

    def test_shadow_ledger_trades_model_vs_market_edge_and_settles_once(self):
        rows = [
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "snapshot_ts": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "model_probability": 0.70,
                "market_probability": 0.90,
                "market_bid": 0.89,
                "market_ask": 0.91,
                "resolved": True,
                "outcome": 0,
                "resolved_ts": datetime(2026, 7, 3, tzinfo=timezone.utc),
            },
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "snapshot_ts": datetime(2026, 7, 2, tzinfo=timezone.utc),
                "model_probability": 0.72,
                "market_probability": 0.88,
                "market_bid": 0.87,
                "market_ask": 0.89,
                "resolved": True,
                "outcome": 0,
                "resolved_ts": datetime(2026, 7, 3, tzinfo=timezone.utc),
            },
        ]

        account = simulate_shadow_mark_to_market_account(
            rows,
            starting_cash=100.0,
            target_contracts=1.0,
        )

        self.assertEqual(account["strategy"], "edge_shadow_ledger")
        self.assertEqual(account["n_trades"], 1)
        self.assertEqual(account["n_settlements"], 1)
        self.assertEqual(account["trades"][0]["side"], NO)
        self.assertEqual(account["settlements"][0]["settled_contracts"], 1)
        self.assertEqual(account["n_open_positions"], 0)
        self.assertAlmostEqual(account["account_value"], 100.89)

    def test_market_follow_baseline_trades_market_favored_side(self):
        rows = [
            {
                "platform": "Polymarket",
                "ident": "MKT",
                "snapshot_ts": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "market_probability": 0.62,
                "market_bid": 0.60,
                "market_ask": 0.64,
                "resolved": False,
            },
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "snapshot_ts": datetime(2026, 7, 1, 1, tzinfo=timezone.utc),
                "market_probability": 0.30,
                "market_bid": 0.29,
                "market_ask": 0.31,
                "resolved": False,
            },
        ]

        account = simulate_market_follow_mark_to_market_account(
            rows,
            starting_cash=100.0,
            latest_quotes={
                ("Polymarket", "MKT"): {"market_bid": 0.61, "market_ask": 0.65},
                ("Kalshi", "KXTEST"): {"market_bid": 0.31, "market_ask": 0.33},
            },
        )

        self.assertEqual(account["strategy"], "market_follow_baseline")
        self.assertEqual(account["n_trades"], 2)
        self.assertEqual(account["trades"][0]["side"], YES)
        self.assertEqual(account["trades"][1]["side"], NO)
        self.assertEqual(account["n_open_positions"], 2)
        self.assertGreater(account["liquidation_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
