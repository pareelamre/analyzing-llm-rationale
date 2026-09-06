import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.execution import ExecutionBlocked, submit_authorized_command
from analyzing_llm_rationale.twin.mandates import Mandate, approve, revoke


class TwinExecutionTests(unittest.TestCase):
    def test_revoked_or_disabled_live_mandates_make_no_write(self):
        now, writes = datetime(2025, 1, 1, tzinfo=timezone.utc), []
        active = approve(Mandate("mandate-001", "owner", "scope", "strategy", now + timedelta(days=1)), owner_id="owner")
        self.assertEqual(submit_authorized_command(active, now=now, live_enabled=False, submit=lambda: writes.append("shadow")), None)
        live = approve(Mandate("mandate-002", "owner", "scope", "strategy", now + timedelta(days=1), live=True, readiness_hash="ready"), owner_id="owner", readiness_hash="ready")
        with self.assertRaises(ExecutionBlocked):
            submit_authorized_command(live, now=now, live_enabled=False, submit=lambda: writes.append("live"))
        with self.assertRaises(ExecutionBlocked):
            submit_authorized_command(revoke(active, owner_id="owner"), now=now, live_enabled=False, submit=lambda: writes.append("bad"))
        self.assertEqual(writes, ["shadow"])
