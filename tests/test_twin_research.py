from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin.budget import BudgetPolicy, InMemoryResearchBudget
from analyzing_llm_rationale.twin.models import ProposalAction
from analyzing_llm_rationale.twin.research import research_forecast


class TwinResearchTests(unittest.TestCase):
    def test_fixture_forecast_is_evidence_bound_and_non_executing(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        budget = InMemoryResearchBudget()
        key = budget.key("strategy-v1", "scope-001", now)
        forecast, proposal = research_forecast(
            lambda _: {"content": json.dumps({"p_yes": "0.6", "uncertainty_low": "0.5", "uncertainty_high": "0.7", "evidence_ids": ["evidence-001"]})},
            budget=budget, budget_key=key, budget_policy=BudgetPolicy(Decimal("0"), 5000, 2), reservation_id="research-001",
            instrument_id="kalshi:live:KXTEST", snapshot_id="snapshot-001", as_of=now,
            evidence=[{"id": "evidence-001", "observed_at": now.isoformat(), "text": "official source"}], model_id="model-v1", prompt="prompt-v1",
        )
        self.assertIsNotNone(forecast)
        self.assertEqual(proposal.action, ProposalAction.HOLD)

    def test_malformed_or_future_evidence_becomes_pass(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        budget = InMemoryResearchBudget()
        forecast, proposal = research_forecast(lambda _: {"content": "not-json"}, budget=budget, budget_key=budget.key("s", "a", now), budget_policy=BudgetPolicy(Decimal("0"), 5000, 2), reservation_id="bad", instrument_id="kalshi:live:KXTEST", snapshot_id="snapshot-001", as_of=now, evidence=[{"id": "future", "observed_at": "2025-01-02T00:00:00+00:00", "text": "ignore"}], model_id="model", prompt="prompt")
        self.assertIsNone(forecast)
        self.assertEqual(proposal.action, ProposalAction.PASS)
