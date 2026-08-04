from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.accounting import (
    NO,
    YES,
    PredictionMarketAccount,
    simulate_mark_to_market_account,
    simulate_market_follow_mark_to_market_account,
    simulate_shadow_mark_to_market_account,
    simulate_validated_kelly_account,
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


class ValidatedKellyAccountTests(unittest.TestCase):
    def _row(self, *, platform="Polymarket", ident, model_p, market_p, calibrated_p=None,
              validated=True, outcome, ts, resolved_ts=None, bid_ask_spread=0.02):
        row = {
            "platform": platform,
            "ident": ident,
            "snapshot_ts": ts,
            "model_probability": model_p,
            "market_probability": market_p,
            "market_bid": max(0.0, market_p - bid_ask_spread / 2),
            "market_ask": min(1.0, market_p + bid_ask_spread / 2),
            "resolved": True,
            "outcome": outcome,
            # Resolution must happen strictly after the snapshot, or the
            # settlement event (priority 0) and this row's trade event
            # (priority 1) land at the identical timestamp and the trade is
            # silently skipped (market already in settled_markets) before it
            # ever reaches the edge/validation checks.
            "resolved_ts": resolved_ts or (ts + timedelta(days=1)),
            "_edge_validated": validated,
        }
        if calibrated_p is not None:
            row["calibrated_model_probability"] = calibrated_p
        return row

    def test_skips_rows_without_edge_validated_by_default(self):
        rows = [
            self._row(ident="m1", model_p=0.7, market_p=0.5, calibrated_p=0.7,
                      validated=False, outcome=1, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(rows, starting_cash=10_000.0)
        self.assertEqual(account["n_trades"], 0)
        self.assertEqual(account["n_skipped_not_validated"], 1)
        self.assertEqual(account["account_value"], 10_000.0)

    def test_trades_validated_row_on_model_implied_side(self):
        rows = [
            self._row(ident="m1", model_p=0.7, market_p=0.5, calibrated_p=0.7,
                      outcome=1, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(rows, starting_cash=10_000.0)
        self.assertEqual(account["n_trades"], 1)
        self.assertEqual(account["trades"][0]["side"], YES)  # model_p > market_p -> YES
        self.assertGreater(account["account_value"], 10_000.0)  # won

    def test_fade_trades_the_opposite_side(self):
        # model says YES (0.7 > market's 0.5), fade should buy NO instead.
        rows = [
            self._row(ident="m1", model_p=0.7, market_p=0.5, calibrated_p=0.3,
                      outcome=0, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(
            rows, starting_cash=10_000.0, fade=True,
        )
        self.assertEqual(account["n_trades"], 1)
        self.assertEqual(account["trades"][0]["side"], NO)
        self.assertEqual(account["strategy"], "fade_kelly_ledger")
        self.assertGreater(account["account_value"], 10_000.0)  # NO won

    def test_max_concentration_caps_stake_regardless_of_kelly_output(self):
        # calibrated_p=0.95 vs market 0.5 implies a huge full-Kelly fraction;
        # max_concentration must still bound the actual dollar stake.
        rows = [
            self._row(ident="m1", model_p=0.95, market_p=0.5, calibrated_p=0.95,
                      outcome=1, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(
            rows, starting_cash=10_000.0, kelly_fraction=1.0, max_concentration=0.10,
        )
        stake = account["trades"][0]["quantity"] * account["trades"][0]["price"]
        self.assertLessEqual(stake, 10_000.0 * 0.10 + 1e-6)

    def test_fixed_fraction_sizing_ignores_model_edge_size(self):
        rows = [
            self._row(ident="m1", model_p=0.55, market_p=0.5, calibrated_p=0.55,
                      outcome=1, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(
            rows, starting_cash=10_000.0, fixed_fraction=0.01,
            max_concentration=0.01, require_validated=False, follow_model_call=True,
        )
        stake = account["trades"][0]["quantity"] * account["trades"][0]["price"]
        self.assertLessEqual(stake, 100.0 + 1e-6)
        self.assertEqual(account["fixed_fraction"], 0.01)

    def test_drawdown_limit_reserves_cash_for_all_open_positions(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rows = [
            self._row(
                ident=f"m{i}", model_p=0.99, market_p=0.5, calibrated_p=0.99,
                outcome=0, ts=base + timedelta(days=2 * i),
                resolved_ts=base + timedelta(days=2 * i + 1),
            )
            for i in range(10)
        ]
        account = simulate_validated_kelly_account(
            rows, starting_cash=10_000.0, kelly_fraction=1.0,
            max_concentration=0.05, max_drawdown=0.30,
        )
        values = [point["account_value"] for point in account["value_curve"]]
        self.assertGreaterEqual(min(values), 7_000.0 - 1e-4)
        self.assertGreater(account["n_skipped_drawdown_limit"], 0)

    def test_open_kelly_ledger_appends_live_mark_to_market_curve_point(self):
        rows = [{
            "platform": "Polymarket",
            "ident": "open-market",
            "snapshot_ts": datetime(2026, 7, 1, tzinfo=timezone.utc),
            "model_probability": 0.70,
            "calibrated_model_probability": 0.70,
            "market_probability": 0.50,
            "market_bid": 0.49,
            "market_ask": 0.51,
            "resolved": False,
            "_edge_validated": True,
        }]
        account = simulate_validated_kelly_account(
            rows,
            latest_quotes={
                ("Polymarket", "open-market"): {
                    "market_bid": 0.59,
                    "market_ask": 0.61,
                },
            },
            starting_cash=10_000.0,
        )

        self.assertEqual(account["n_open_positions"], 1)
        self.assertEqual(account["value_curve"][-1]["event_type"], "mark_to_market")
        self.assertEqual(account["value_curve"][-1]["account_value"], account["account_value"])

    def test_skips_prices_outside_the_default_band(self):
        # market_p=0.05 -> ask for the model's YES-implied side sits near 0.06,
        # well outside the default [0.20, 0.80] band -- must be skipped even
        # though it's flagged validated.
        rows = [
            self._row(ident="m1", model_p=0.5, market_p=0.05, calibrated_p=0.9,
                      outcome=1, ts=datetime(2026, 7, 1, tzinfo=timezone.utc))
        ]
        account = simulate_validated_kelly_account(rows, starting_cash=10_000.0)
        self.assertEqual(account["n_trades"], 0)
        self.assertEqual(account["account_value"], 10_000.0)

    def test_settles_once_per_market_across_duplicate_snapshots(self):
        base = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rows = [
            self._row(ident="m1", model_p=0.7, market_p=0.5, calibrated_p=0.7,
                      outcome=1, ts=base, resolved_ts=base),
            self._row(ident="m1", model_p=0.72, market_p=0.52, calibrated_p=0.72,
                      outcome=1, ts=base, resolved_ts=base),
        ]
        account = simulate_validated_kelly_account(rows, starting_cash=10_000.0)
        self.assertEqual(account["n_settlements"], 1)


if __name__ == "__main__":
    unittest.main()
