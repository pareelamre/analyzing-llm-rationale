from decimal import Decimal
import unittest

from analyzing_llm_rationale.twin.risk import RiskLimits, size_binary_entry


class TwinRiskTests(unittest.TestCase):
    def test_costs_limits_and_rounding_bind_size(self):
        result = size_binary_entry(probability=Decimal(".70"), ask=Decimal(".50"), fee_per_share=Decimal(".01"), slippage_per_share=Decimal(".01"), available_cash=Decimal("100"), current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"), drawdown=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("1"), limits=RiskLimits(Decimal(".1"), Decimal("10"), Decimal("20"), Decimal("20"), Decimal(".2")))
        self.assertGreater(result.quantity, 0)
        self.assertLessEqual(result.cash, Decimal("10"))

    def test_missing_edge_drawdown_or_minimum_cap_passes(self):
        base = dict(probability=Decimal(".6"), ask=Decimal(".5"), fee_per_share=Decimal("0"), slippage_per_share=Decimal("0"), available_cash=Decimal("1"), current_market_loss=Decimal("0"), current_cluster_loss=Decimal("0"), tick_size=Decimal(".01"), min_quantity=Decimal("2"), limits=RiskLimits(Decimal(".1"), Decimal("1"), Decimal("1"), Decimal("1"), Decimal(".2")))
        self.assertEqual(size_binary_entry(**base, drawdown=Decimal(".2")).reason, "drawdown_limit")
        self.assertEqual(size_binary_entry(**base, drawdown=Decimal("0")).reason, "minimum_size_exceeds_cap")
