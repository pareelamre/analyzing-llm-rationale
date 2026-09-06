import unittest
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
