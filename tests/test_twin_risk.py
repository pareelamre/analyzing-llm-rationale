from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import (
    AccountSnapshot,
    CalibrationResult,
    Completeness,
    MarketSnapshot,
    RiskExposure,
    RiskLimits,
    calibrate_probability,
    evaluate_binary_candidate,
    size_binary_entry,
    size_reduce_only,
    sorted_candidate_ids,
)
from analyzing_llm_rationale.twin.store import AccountProjection

NOW = datetime(2025, 2, 1, tzinfo=timezone.utc)


def calibration(*, low: str = ".65", high: str = ".75") -> CalibrationResult:
    return CalibrationResult(
        Decimal(low), 30, "calibration-hash", mean_probability=Decimal(".7"),
        lower_bound=Decimal(low), upper_bound=Decimal(high), cutoff=NOW,
    )


def account(*, divergence: bool = False, received_at: datetime | None = None) -> AccountSnapshot:
    return AccountSnapshot(
        scope_id="scope-001", generation=4, received_at=received_at or NOW - timedelta(seconds=5),
        completeness=Completeness.COMPLETE, available_cash=Decimal("100"), total_cash=Decimal("100"),
        reserved_cash=Decimal("0"), settled_cash=Decimal("100"), holdings=(),
        position_basis=Decimal("0"), fees_paid=Decimal("0"),
        conservative_liquidation_value=Decimal("100"), positions=(), orders=(), fills=(), settlements=(),
        external_activity_ids=("manual-order",) if divergence else (), divergence=divergence,
        drift_reasons=("external_activity",) if divergence else (),
    )


def market(*, received_at: datetime | None = None) -> MarketSnapshot:
    received = received_at or NOW - timedelta(seconds=1)
    return MarketSnapshot(
        id="snapshot-001", instrument_id="instrument-001", venue_at=received - timedelta(seconds=1),
        received_at=received, sequence=9, source="venue-rest",
        complete=Completeness.COMPLETE, stale_after_seconds=10, yes_bid=Decimal(".39"),
        yes_ask=Decimal(".40"), no_bid=Decimal(".59"), no_ask=Decimal(".60"),
        fee_version="fee-v1", created_at=NOW - timedelta(seconds=1),
    )


def projection(*, cash: str = "100", reserved_cash: str = "0", reserved_loss: str = "0") -> AccountProjection:
    return AccountProjection(
        "scope-001", 1, Decimal(cash), Decimal("100"), revision=7,
        reserved_cash=Decimal(reserved_cash), reserved_max_loss=Decimal(reserved_loss),
    )


def limits(**changes: Decimal) -> RiskLimits:
    values = {
        "kelly_fraction": Decimal(".1"), "max_order_cash": Decimal("10"),
        "max_market_loss": Decimal("10"), "max_cluster_loss": Decimal("15"),
        "max_drawdown": Decimal(".2"), "max_total_loss": Decimal("20"),
        "max_trailing_additions": Decimal("10"), "max_daily_realized_loss": Decimal("5"),
    }
    values.update(changes)
    return RiskLimits(**values)


def evaluate(**changes):
    values = {
        "action": "BUY_YES", "instrument_id": "instrument-001", "cluster_id": "cluster-001",
        "venue": "kalshi", "market_snapshot": market(), "account_snapshot": account(),
        "account_projection": projection(), "exposures": (), "limits": limits(), "now": NOW,
        "calibration": calibration(), "fee_per_share": Decimal(".01"),
        "slippage_per_share": Decimal(".01"), "fee_version": "fee-v1",
        "available_depth": Decimal("100"),
        "cash_increment": Decimal(".01"), "tick_size": Decimal(".01"),
        "min_quantity": Decimal("1"), "trailing_additions": Decimal("0"),
        "realized_losses": Decimal("0"), "peak_equity": Decimal("100"),
        "current_equity": Decimal("100"),
    }
    values.update(changes)
    return evaluate_binary_candidate(**values)


class TwinRiskTests(unittest.TestCase):
    def test_costs_depth_limits_and_currency_rounding_bind_size(self):
        result = size_binary_entry(
            probability=Decimal(".70"), ask=Decimal(".50"), fee_per_share=Decimal(".013"),
            slippage_per_share=Decimal(".011"), available_cash=Decimal("100"),
            current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"),
            drawdown=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("1"),
            available_depth=Decimal("3.5"), cash_increment=Decimal(".01"),
            limits=RiskLimits(Decimal("1"), Decimal("10"), Decimal("20"), Decimal("20"), Decimal(".2")),
        )
        self.assertEqual(result.quantity, Decimal("3"))
        self.assertEqual(result.cash, Decimal("1.58"))
        self.assertEqual(result.cash_delta, Decimal("-1.58"))

    def test_fixed_bin_calibration_is_prospective_stratified_and_independent(self):
        rows = []
        for index in range(30):
            rows.append({
                "id": f"forecast-{index:02d}", "instrument_id": f"instrument-{index:02d}",
                "cluster_id": f"cluster-{index:02d}", "probability": ".64",
                "outcome": 1 if index < 24 else 0, "forecast_at": (NOW - timedelta(days=10)).isoformat(),
                "resolved_at": (NOW - timedelta(days=2)).isoformat(), "model_hash": "model-a",
                "prompt_hash": "prompt-a", "category_family": "politics",
            })
        rows.extend([
            {**rows[0], "id": "later-duplicate", "outcome": 0, "forecast_at": (NOW - timedelta(days=9)).isoformat()},
            {**rows[1], "id": "future-resolution", "resolved_at": (NOW + timedelta(days=1)).isoformat()},
            {**rows[2], "id": "wrong-model", "model_hash": "model-b"},
            {**rows[3], "id": "outcome-known-first", "forecast_at": (NOW - timedelta(days=1)).isoformat()},
        ])
        result = calibrate_probability(
            Decimal(".61"), list(reversed(rows)), as_of=NOW, model_hash="model-a",
            prompt_hash="prompt-a", category_family="politics",
        )
        repeat = calibrate_probability(
            Decimal(".61"), rows, as_of=NOW, model_hash="model-a",
            prompt_hash="prompt-a", category_family="politics",
        )
        self.assertEqual(result, repeat)
        self.assertEqual(result.sample_size, 30)
        self.assertEqual(result.mean_probability, Decimal("25") / Decimal("32"))
        self.assertEqual(result.training_ids[0], "forecast-00")
        self.assertGreater(result.probability, Decimal(".6"))
        self.assertEqual(result.calibration_version, "calibration_v1")

    def test_calibration_blocks_small_samples_and_no_side_uses_upper_bound(self):
        rows = [{
            "id": f"f-{index}", "instrument_id": f"i-{index}", "cluster_id": f"c-{index}",
            "probability": ".24", "outcome": 0, "forecast_at": (NOW - timedelta(days=3)).isoformat(),
            "resolved_at": (NOW - timedelta(days=1)).isoformat(),
        } for index in range(30)]
        self.assertEqual(
            calibrate_probability(Decimal(".24"), rows[:29], as_of=NOW).reason,
            "insufficient_calibration_sample",
        )
        no_result = calibrate_probability(Decimal(".24"), rows, as_of=NOW, outcome_side="NO")
        self.assertEqual(no_result.probability, Decimal("1") - no_result.upper_bound)

    def test_yes_and_no_entry_use_conservative_interval_and_capture_versions(self):
        yes = evaluate()
        no = evaluate(action="BUY_NO", calibration=calibration(low=".10", high=".30"))
        self.assertGreater(yes.quantity, 0)
        self.assertGreater(no.quantity, 0)
        self.assertEqual(yes.expected_account_revision, 7)
        self.assertEqual(yes.account_generation, 4)
        self.assertEqual(yes.market_snapshot_id, "snapshot-001")
        self.assertEqual(yes.calibration_hash, "calibration-hash")
        self.assertEqual(yes.reservation_preconditions.market_version, "snapshot-001")

    def test_inventory_orders_and_reservations_bind_market_cluster_and_total_caps(self):
        exposures = (
            RiskExposure("instrument-001", "cluster-001", "kalshi", Decimal("4"), "inventory"),
            RiskExposure("instrument-002", "cluster-001", "polymarket", Decimal("7"), "open_order"),
            RiskExposure("instrument-003", "cluster-002", "kalshi", Decimal("2"), "local_reservation"),
        )
        result = evaluate(
            exposures=exposures, account_projection=projection(reserved_cash="2", reserved_loss="2"),
            limits=limits(max_cluster_loss=Decimal("12"), max_total_loss=Decimal("14")),
        )
        self.assertEqual(result.quantity, Decimal("2"))
        self.assertLessEqual(result.max_loss, Decimal("1"))

    def test_reduce_only_accounts_for_pending_sells_partial_fills_and_depth(self):
        result = evaluate(
            action="SELL_YES", calibration=None, requested_quantity=Decimal("10"),
            held_quantity=Decimal("5"), pending_sell_quantity=Decimal("2"),
            available_depth=Decimal("2.5"),
        )
        self.assertEqual(result.quantity, Decimal("2"))
        self.assertEqual(result.cash, Decimal("0"))
        self.assertEqual(result.cash_delta, Decimal(".74"))
        self.assertEqual(
            size_reduce_only(
                held_quantity=Decimal("3"), pending_sell_quantity=Decimal("3"),
                requested_quantity=Decimal("1"), min_quantity=Decimal("1"),
            ).reason,
            "insufficient_inventory",
        )

    def test_external_activity_staleness_and_incomplete_portfolio_block(self):
        self.assertEqual(evaluate(account_snapshot=account(divergence=True)).reason, "account_reconciliation_blocked")
        self.assertEqual(
            evaluate(market_snapshot=market(received_at=NOW - timedelta(seconds=11))).reason,
            "stale_market_snapshot",
        )
        self.assertEqual(evaluate(exposures=None).reason, "incomplete_portfolio_state")
        self.assertEqual(
            evaluate(exposures=(), account_projection=projection(reserved_loss="1", reserved_cash="1")).reason,
            "incomplete_portfolio_state",
        )

    def test_missing_calibration_cost_depth_cash_and_uncertainty_block(self):
        self.assertEqual(evaluate(calibration=None).reason, "missing_calibration")
        self.assertEqual(
            evaluate(calibration=CalibrationResult(Decimal(".6"), 30, "hash")).reason,
            "missing_calibration",
        )
        self.assertEqual(evaluate(available_depth=None).reason, "missing_cost_or_depth")
        self.assertEqual(evaluate(fee_per_share=None).reason, "missing_cost_or_depth")
        self.assertEqual(evaluate(fee_version="fee-v0").reason, "stale_fee_schedule")
        self.assertEqual(
            evaluate(
                account_projection=projection(cash="0"),
                account_snapshot=replace(account(), available_cash=Decimal("0")),
            ).reason,
            "minimum_size_exceeds_cap",
        )

    def test_trailing_realized_and_peak_equity_limits_are_separate(self):
        self.assertEqual(evaluate(trailing_additions=Decimal("10")).reason, "trailing_additions_limit")
        self.assertEqual(evaluate(realized_losses=Decimal("5")).reason, "realized_loss_limit")
        self.assertEqual(evaluate(current_equity=Decimal("80")).reason, "drawdown_limit")
        self.assertEqual(evaluate(peak_equity=Decimal("0"), current_equity=Decimal("0")).reason, "zero_bankroll")

    def test_minimum_order_and_deterministic_priority_are_stable(self):
        self.assertEqual(evaluate(min_quantity=Decimal("1000")).reason, "minimum_size_exceeds_cap")
        self.assertEqual(
            sorted_candidate_ids([
                {"id": "b", "net_edge": ".1"}, {"id": "a", "net_edge": ".1"},
                {"id": "c", "net_edge": ".2"},
            ]),
            ("c", "a", "b"),
        )


if __name__ == "__main__":
    unittest.main()
