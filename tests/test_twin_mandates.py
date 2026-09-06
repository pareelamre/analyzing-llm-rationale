import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.mandates import Mandate, approve, revoke


class TwinMandateTests(unittest.TestCase):
    def test_owner_approval_expiry_and_revocation(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        draft = Mandate("mandate-001", "owner-001", "scope-001", "strategy-v1", now + timedelta(days=1))
        active = approve(draft, owner_id="owner-001")
        self.assertTrue(active.active(now=now))
        self.assertFalse(revoke(active, owner_id="owner-001").active(now=now))
        with self.assertRaises(PermissionError):
            approve(draft, owner_id="other")

    def test_approval_binds_budget_scope_and_readiness_hash(self):
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        draft = Mandate(
            "mandate-live", "owner", "scope", "strategy", now + timedelta(days=1), live=True,
            account_epoch=2, venue="kalshi", max_capital="10", max_loss="10", readiness_hash="ready-v1",
        )
        active = approve(draft, owner_id="owner", readiness_hash="ready-v1")
        self.assertTrue(active.active(now=now))
        self.assertFalse(replace(active, max_capital="11").active(now=now))
        self.assertEqual(revoke(active, owner_id="owner"), revoke(revoke(active, owner_id="owner"), owner_id="owner"))
