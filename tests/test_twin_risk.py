import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin.risk import (
    RiskLimits,
    calibrate_probability,
    size_binary_entry,
    size_reduce_only,
    sorted_candidate_ids,
)


class TwinRiskTests(unittest.TestCase):
    def test_costs_limits_and_rounding_bind_size(self):
        result = size_binary_entry(probability=Decimal(".70"), ask=Decimal(".50"), fee_per_share=Decimal(".01"), slippage_per_share=Decimal(".01"), available_cash=Decimal("100"), current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"), drawdown=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("1"), limits=RiskLimits(Decimal(".1"), Decimal("10"), Decimal("20"), Decimal("20"), Decimal(".2")))
        self.assertGreater(result.quantity, 0)
        self.assertLessEqual(result.cash, Decimal("10"))

    def test_missing_edge_drawdown_or_minimum_cap_passes(self):
        base = dict(probability=Decimal(".6"), ask=Decimal(".5"), fee_per_share=Decimal("0"), slippage_per_share=Decimal("0"), available_cash=Decimal("1"), current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("2"), limits=RiskLimits(Decimal(".1"), Decimal("1"), Decimal("1"), Decimal("1"), Decimal(".2")))
        self.assertEqual(size_binary_entry(**base, drawdown=Decimal(".2")).reason, "drawdown_limit")
        self.assertEqual(size_binary_entry(**base, drawdown=Decimal("0")).reason, "minimum_size_exceeds_cap")

    def test_lot_rounding_and_cumulative_cash_loss_are_exact(self):
        limits = RiskLimits(Decimal("1"), Decimal("10"), Decimal("10"), Decimal("10"), Decimal(".5"))
        result = size_binary_entry(
            probability=Decimal(".9"), ask=Decimal(".50"), fee_per_share=Decimal(".02"),
            slippage_per_share=Decimal(".03"), available_cash=Decimal("100"),
            current_market_loss=Decimal("4.75"), current_cluster_loss=Decimal("0"), drawdown=Decimal("0"),
            tick_size=Decimal(".01"), min_quantity=Decimal("5"), limits=limits,
        )
        self.assertEqual(result.quantity, Decimal("5"))
        self.assertEqual(result.cash, Decimal("2.75"))
        self.assertEqual(result.max_loss, result.cash)
        blocked = size_binary_entry(
            probability=Decimal(".9"), ask=Decimal(".503"), fee_per_share=Decimal("0"),
            slippage_per_share=Decimal("0"), available_cash=Decimal("100"), current_market_loss=Decimal("0"),
            current_cluster_loss=Decimal("0"), drawdown=Decimal("0"), tick_size=Decimal(".01"),
            min_quantity=Decimal("1"), limits=limits,
        )
        self.assertEqual(blocked.reason, "invalid_market_input")

    def test_calibration_excludes_future_outcomes_and_reduce_only_never_flips(self):
        as_of = datetime(2025, 2, 1, tzinfo=timezone.utc)
        resolved = as_of - timedelta(days=1)
        observations = [
            {"id": f"old-{index}", "probability": ".62", "outcome": 1, "resolved_at": resolved.isoformat()}
            for index in range(20)
        ] + [{"id": "future", "probability": ".62", "outcome": 0, "resolved_at": (as_of + timedelta(days=1)).isoformat()}]
        calibration = calibrate_probability(Decimal(".61"), observations, as_of=as_of)
        self.assertEqual(calibration.sample_size, 20)
        self.assertGreater(calibration.probability, Decimal(".5"))
        self.assertEqual(size_reduce_only(held_quantity=Decimal("3"), requested_quantity=Decimal("9"), min_quantity=Decimal("1")).quantity, Decimal("3"))
        self.assertEqual(sorted_candidate_ids([{"id": "b", "net_edge": ".1"}, {"id": "a", "net_edge": ".1"}]), ("a", "b"))
