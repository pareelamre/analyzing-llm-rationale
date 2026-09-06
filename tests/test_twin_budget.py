from __future__ import annotations

import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin.budget import (
    BudgetExceeded,
    BudgetPolicy,
    InMemoryResearchBudget,
    ModelPrice,
    PriceUnavailable,
    estimate_request_cost,
)


class TwinBudgetTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryResearchBudget()
        self.key = self.store.key("strategy-v1", "scope-001", datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.policy = BudgetPolicy(Decimal("1"), 100, 2)

    def test_two_workers_can_only_reserve_final_allowance_once(self):
        outcomes = []
        def reserve(name):
            try:
                self.store.reserve(name, key=self.key, estimated_usd=Decimal("1"), estimated_tokens=50, policy=self.policy)
                outcomes.append("reserved")
            except BudgetExceeded:
                outcomes.append("blocked")
        threads = [threading.Thread(target=reserve, args=(f"request-{i}",)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(outcomes), ["blocked", "reserved"])

    def test_timeout_remains_uncertain_and_actual_usage_can_exceed_estimate(self):
        self.store.reserve("timeout", key=self.key, estimated_usd=Decimal("0.4"), estimated_tokens=20, policy=self.policy)
        unknown = self.store.reconcile("timeout", actual_usd=None, actual_tokens=None)
        self.assertEqual(unknown.uncertain_usd, Decimal("0.4"))
        self.store.reserve("actual", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy)
        actual = self.store.reconcile("actual", actual_usd=Decimal("0.3"), actual_tokens=12)
        self.assertEqual(actual.actual_usd, Decimal("0.3"))
        self.assertEqual(actual.actual_tokens, 12)

    def test_idempotency_and_utc_day_boundary(self):
        first = self.store.reserve("same", key=self.key, estimated_usd=Decimal("0"), estimated_tokens=10, policy=self.policy)
        self.assertEqual(first, self.store.reserve("same", key=self.key, estimated_usd=Decimal("0.9"), estimated_tokens=90, policy=self.policy))
        next_key = self.store.key("strategy-v1", "scope-001", datetime(2025, 1, 2, tzinfo=timezone.utc))
        self.store.reserve("next-day", key=next_key, estimated_usd=Decimal("1"), estimated_tokens=100, policy=self.policy)
        self.assertNotEqual(self.key, next_key)

    def test_missing_price_blocks_paid_research_but_explicit_zero_cost_is_limited_by_tokens(self):
        with self.assertRaises(PriceUnavailable):
            estimate_request_cost(
                input_tokens=10, output_tokens=10,
                price=ModelPrice(None, None), require_usd_ceiling=True,
            )
        self.assertEqual(
            estimate_request_cost(
                input_tokens=10, output_tokens=10,
                price=ModelPrice(Decimal("0"), Decimal("0")), require_usd_ceiling=True,
            ),
            Decimal("0"),
        )


if __name__ == "__main__":
    unittest.main()
