from __future__ import annotations

import threading
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from analyzing_llm_rationale.twin.budget import (
    BudgetAlreadyClaimed,
    BudgetExceeded,
    BudgetPolicy,
    DatastoreResearchBudget,
    InMemoryResearchBudget,
    ModelPrice,
    PriceUnavailable,
    call_with_budget,
    estimate_request_cost,
)
from analyzing_llm_rationale.twin.research_gateway import load_research_runtime_policy

ROOT = Path(__file__).resolve().parents[1]


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
        self.assertEqual(first, self.store.reserve("same", key=self.key, estimated_usd=Decimal("0"), estimated_tokens=10, policy=self.policy))
        with self.assertRaises(ValueError):
            self.store.reserve("same", key=self.key, estimated_usd=Decimal("0.9"), estimated_tokens=90, policy=self.policy)
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

    def test_provider_failure_or_missing_usage_remains_uncertain(self):
        with self.assertRaises(RuntimeError):
            call_with_budget(
                self.store, "failed", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10,
                policy=self.policy, operation=lambda: (_ for _ in ()).throw(RuntimeError("timeout")),
            )
        self.assertEqual(self.store.usage(self.key).uncertain_usd, Decimal("0.2"))

    def test_uncertain_tokens_and_cash_cannot_be_spent_again(self):
        self.store.reserve("unknown", key=self.key, estimated_usd=Decimal("0.8"), estimated_tokens=90, policy=self.policy)
        usage = self.store.reconcile("unknown", key=self.key, actual_usd=None, actual_tokens=None)
        self.assertEqual(usage.uncertain_tokens, 90)
        for dollars, tokens in [("0.3", 0), ("0", 11)]:
            with self.subTest(dollars=dollars, tokens=tokens), self.assertRaises(BudgetExceeded):
                self.store.reserve("next", key=self.key, estimated_usd=Decimal(dollars), estimated_tokens=tokens, policy=self.policy)

    def test_duplicate_delivery_calls_provider_once_including_timeout(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                store = InMemoryResearchBudget()
                calls = []
                outcomes = []

                def operation(calls=calls, fail=fail):
                    calls.append(1)
                    if fail:
                        raise TimeoutError("response lost")
                    return {"usage": {"cost_usd": "0.2", "total_tokens": 10}}

                def invoke(store=store, outcomes=outcomes, operation=operation):
                    try:
                        call_with_budget(store, "same", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy, operation=operation)
                        outcomes.append("completed")
                    except BudgetAlreadyClaimed:
                        outcomes.append("duplicate")
                    except TimeoutError:
                        outcomes.append("timeout")

                threads = [threading.Thread(target=invoke) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(len(calls), 1)
                self.assertCountEqual(outcomes, ["duplicate", "timeout" if fail else "completed"])
                invoke()
                self.assertEqual(len(calls), 1)

    def test_invalid_usage_keeps_entire_reservation_uncertain(self):
        for usage in [{"cost_usd": "-1", "total_tokens": 10}, {"cost_usd": "NaN", "total_tokens": 10}, {"cost_usd": "0", "total_tokens": True}, {"cost_usd": "0", "total_tokens": -1}]:
            with self.subTest(usage=usage):
                store = InMemoryResearchBudget()
                with self.assertRaises(ValueError):
                    call_with_budget(store, "bad", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy, operation=lambda usage=usage: {"usage": usage})
                self.assertEqual(store.usage(self.key).uncertain_usd, Decimal("0.2"))
                self.assertEqual(store.usage(self.key).uncertain_tokens, 10)

    def test_partial_usage_above_estimate_is_retained(self):
        self.store.reserve("partial", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy)
        usage = self.store.reconcile("partial", key=self.key, actual_usd=None, actual_tokens=120)
        self.assertEqual(usage.uncertain_tokens, 120)
        with self.assertRaises(BudgetExceeded):
            self.store.reserve("next", key=self.key, estimated_usd=Decimal("0"), estimated_tokens=1, policy=self.policy)

    def test_malformed_partial_usage_preserves_known_overrun(self):
        for usage, cost, tokens in [({"cost_usd": "1.5", "total_tokens": "bad"}, "1.5", 10), ({"cost_usd": "bad", "total_tokens": 120}, "0.2", 120)]:
            with self.subTest(usage=usage):
                store = InMemoryResearchBudget()
                with self.assertRaises(ValueError):
                    call_with_budget(store, "bad", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy, operation=lambda usage=usage: {"usage": usage})
                self.assertEqual(store.usage(self.key).uncertain_usd, Decimal(cost))
                self.assertEqual(store.usage(self.key).uncertain_tokens, tokens)
                with self.assertRaises(BudgetExceeded):
                    store.reserve("next", key=self.key, estimated_usd=Decimal("0.1"), estimated_tokens=1, policy=self.policy)

    def test_invalid_policy_price_or_reservation_cannot_create_capacity(self):
        for amount in ("NaN", "Infinity", "-0.1"):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    BudgetPolicy(Decimal(amount), 100, 2)
                with self.assertRaises(ValueError):
                    self.store.reserve("invalid", key=self.key, estimated_usd=Decimal(amount), estimated_tokens=10, policy=self.policy)
                with self.assertRaises(ValueError):
                    estimate_request_cost(input_tokens=10, output_tokens=10, price=ModelPrice(Decimal(amount), Decimal("0")), require_usd_ceiling=True)

    def test_authoritative_late_usage_replaces_uncertainty_once(self):
        self.store.reserve("late", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=10, policy=self.policy)
        self.store.reconcile("late", key=self.key, actual_usd=None, actual_tokens=None)
        self.store.reconcile("late", key=self.key, actual_usd=Decimal("0.9"), actual_tokens=None)
        usage = self.store.reconcile("late", key=self.key, actual_usd=Decimal("0.9"), actual_tokens=90)
        self.assertEqual(usage.uncertain_usd, 0)
        self.assertEqual(usage.uncertain_tokens, 0)
        self.assertEqual(usage.actual_usd, Decimal("0.9"))
        self.assertEqual(usage.actual_tokens, 90)
        self.assertEqual(usage, self.store.reconcile("late", key=self.key, actual_usd=Decimal("0.9"), actual_tokens=90))
        with self.assertRaises(BudgetExceeded):
            self.store.reserve("next", key=self.key, estimated_usd=Decimal("0.2"), estimated_tokens=1, policy=self.policy)
        with self.assertRaises(ValueError):
            self.store.reconcile("late", key=self.key, actual_usd=Decimal("0.1"), actual_tokens=90)

    def test_stored_usage_cannot_create_spending_capacity(self):
        for field, value in [("actual_usd", "-10"), ("reserved_usd", "NaN"), ("actual_tokens", -100), ("reserved_tokens", 1.5), ("uncertain_tokens", True)]:
            with self.subTest(field=field), self.assertRaises((ValueError, ArithmeticError)):
                DatastoreResearchBudget._usage("test", {"requests": 1, "uncertain_tokens": 0, field: value})

    def test_runtime_policy_loads_one_explicitly_free_bounded_model(self):
        policy = load_research_runtime_policy(
            ROOT / "configs" / "twin.yaml", ROOT / "configs" / "models.yaml",
        )
        self.assertEqual(policy.model_key, "gpt-oss-120b")
        self.assertEqual(policy.model.model_id, "openai/gpt-oss-120b")
        self.assertEqual(policy.model.price, ModelPrice(Decimal("0"), Decimal("0")))
        self.assertEqual(
            (policy.candidates_per_cycle, policy.tool_calls_per_candidate,
             policy.schema_repairs_per_candidate),
            (3, 8, 1),
        )
        self.assertEqual(policy.model.max_request_seconds, 60)
        with mock.patch.dict("os.environ", {policy.api_key_env_var: "fixture-key"}):
            provider = policy.build_provider()
        self.assertEqual(provider.model_name, policy.model.model_id)
        self.assertEqual(provider.request_timeout_s, 60)

    def test_runtime_policy_rejects_missing_price_or_widened_fanout(self):
        source = (ROOT / "configs" / "twin.yaml").read_text(encoding="utf-8")
        cases = (
            (source.replace('input: "0"', "input: null"), "price"),
            (source.replace("candidates_per_cycle: 3", "candidates_per_cycle: 4"), "fanout"),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "twin.yaml"
            for value, error in cases:
                with self.subTest(error=error):
                    path.write_text(value, encoding="utf-8")
                    with self.assertRaises((PriceUnavailable, ValueError)):
                        load_research_runtime_policy(path, ROOT / "configs" / "models.yaml")


if __name__ == "__main__":
    unittest.main()
