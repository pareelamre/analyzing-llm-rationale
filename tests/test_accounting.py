from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.accounting import (  # noqa: E402
    MarketQuote,
    PredictionMarketAccount,
    simulate_mark_to_market_account,
    simulate_shadow_mark_to_market_account,
)


class AccountingTests(unittest.TestCase):
    def test_account_value_uses_bid_liquidation_not_entry_price(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(
            platform="Kalshi",
            ident="KXTEST",
            side="YES",
            quantity=10,
            price=0.60,
        )

        snapshot = account.snapshot({
            ("Kalshi", "KXTEST"): MarketQuote(yes_bid=0.48, yes_ask=0.52),
        })

        self.assertEqual(snapshot["cash"], 94.0)
        self.assertEqual(snapshot["liquidation_value"], 4.8)
        self.assertEqual(snapshot["account_value"], 98.8)
        self.assertEqual(snapshot["value_method"], "mark_to_market_bid_liquidation")

    def test_buy_no_nets_yes_and_records_realized_pnl(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(
            platform="Kalshi",
            ident="KXTEST",
            side="YES",
            quantity=10,
            price=0.60,
        )
        fill = account.buy(
            platform="Kalshi",
            ident="KXTEST",
            side="NO",
            quantity=10,
            price=0.35,
            fee=0.17,
        )

        self.assertEqual(fill.settlement_status, "realized")
        self.assertEqual(fill.realized_pairs, 10)
        self.assertEqual(round(fill.realized_pnl, 2), 0.33)
        self.assertEqual(account.open_positions(), [])
        self.assertEqual(round(account.cash, 2), 100.33)

    def test_settlement_records_realized_pnl(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(platform="Kalshi", ident="KXTEST", side="YES", quantity=2, price=0.40)

        settlement = account.settle_market(
            platform="Kalshi",
            ident="KXTEST",
            outcome=1,
            ts=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(settlement["settled_contracts"], 2)
        self.assertEqual(settlement["payout"], 2)
        self.assertEqual(round(settlement["realized_pnl"], 2), 1.20)
        self.assertEqual(round(account.realized_pnl, 2), 1.20)
        self.assertEqual(round(account.cash, 2), 101.20)

    def test_partial_netting_keeps_remaining_original_side(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(platform="Kalshi", ident="KXTEST", side="YES", quantity=10, price=0.60)
        fill = account.buy(platform="Kalshi", ident="KXTEST", side="NO", quantity=6, price=0.35)

        positions = account.open_positions()
        self.assertEqual(fill.settlement_status, "realized")
        self.assertEqual(fill.realized_pairs, 6)
        self.assertEqual(round(fill.realized_pnl, 2), 0.30)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].side, "YES")
        self.assertEqual(positions[0].quantity, 4)
        self.assertEqual(round(positions[0].cost_basis, 2), 2.40)

    def test_zero_bid_position_is_marked_illiquid_at_zero_value(self):
        account = PredictionMarketAccount(starting_cash=100.0)
        account.buy(platform="Kalshi", ident="KXTEST", side="YES", quantity=5, price=0.20)

        snapshot = account.snapshot({
            ("Kalshi", "KXTEST"): MarketQuote(yes_bid=0.0, yes_ask=0.25),
        })

        self.assertEqual(snapshot["liquidation_value"], 0.0)
        self.assertEqual(snapshot["account_value"], 99.0)
        self.assertEqual(snapshot["illiquid_positions"][0]["reason"], "zero_or_missing_bid")

    def test_simulation_rebalances_by_buying_opposite_side_and_marks_to_latest_bid(self):
        rows = [
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "snapshot_ts": datetime(2026, 7, 1, tzinfo=timezone.utc),
                "model_probability": 0.70,
                "market_probability": 0.60,
                "market_bid": 0.59,
                "market_ask": 0.61,
            },
            {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "snapshot_ts": datetime(2026, 7, 2, tzinfo=timezone.utc),
                "model_probability": 0.30,
                "market_probability": 0.55,
                "market_bid": 0.54,
                "market_ask": 0.56,
            },
        ]

        result = simulate_mark_to_market_account(
            rows,
            latest_quotes={
                ("Kalshi", "KXTEST"): MarketQuote(yes_bid=0.54, yes_ask=0.56),
            },
            starting_cash=100.0,
            target_contracts=1.0,
        )

        self.assertEqual(result["n_trades"], 2)
        self.assertEqual(result["trades"][1]["settlement_status"], "realized")
        self.assertEqual(result["open_positions"][0]["side"], "NO")
        self.assertEqual(result["liquidation_value"], 0.44)

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

        result = simulate_shadow_mark_to_market_account(
            rows,
            starting_cash=100.0,
            target_contracts=1.0,
        )

        self.assertEqual(result["strategy"], "edge_shadow_ledger")
        self.assertEqual(result["n_trades"], 1)
        self.assertEqual(result["n_settlements"], 1)
        self.assertEqual(result["trades"][0]["side"], "NO")
        self.assertEqual(result["settlements"][0]["settled_contracts"], 1)
        self.assertEqual(result["n_open_positions"], 0)
        self.assertEqual(round(result["account_value"], 2), 100.89)


if __name__ == "__main__":
    unittest.main()
