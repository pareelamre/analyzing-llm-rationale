import unittest
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.worker import (
    InMemoryWorkerJobs,
    TwinWorker,
    WorkerAuthenticationError,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    require_worker_request,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def job(job_id="job-001", *, kind=WorkerJobKind.RECONCILE, deadline=NOW + timedelta(minutes=1)):
    return WorkerJob(job_id, "scope-001", kind, {"market_id": "market-001"}, deadline)


class TwinWorkerTests(unittest.TestCase):
    def test_private_worker_rejects_public_request(self):
        require_worker_request("valid", expected_token="valid")
        with self.assertRaises(WorkerAuthenticationError):
            require_worker_request(None, expected_token="valid")

    def test_duplicate_delivery_returns_saved_result_and_never_runs_twice(self):
        jobs, calls = InMemoryWorkerJobs(), []
        jobs.add(job())
        worker = TwinWorker(jobs, worker_id="worker-a", reconcile_startup=lambda: True)
        first = worker.handle("job-001", now=NOW, maintain=lambda _: calls.append("maintenance") or {"ok": True}, research=lambda _: {})
        second = worker.handle("job-001", now=NOW, maintain=lambda _: calls.append("again") or {}, research=lambda _: {})
        self.assertEqual(first, {"ok": True})
        self.assertEqual(second, {"ok": True})
        self.assertEqual(calls, ["maintenance"])

    def test_expired_lease_and_startup_reconciliation_are_safe(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job())
        first = jobs.claim("job-001", worker_id="dead", now=NOW, lease_seconds=1)
        self.assertIsNotNone(first)
        second = jobs.claim("job-001", worker_id="new", now=NOW + timedelta(seconds=2))
        self.assertEqual(second.worker_id, "new")
        worker = TwinWorker(jobs, worker_id="new", reconcile_startup=lambda: False)
        self.assertFalse(worker.start())

    def test_priority_and_payload_constraints_keep_maintenance_ahead_of_research(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        jobs.add(job("recovery", kind=WorkerJobKind.RECOVERY))
        self.assertEqual([item.id for item in jobs.due(now=NOW)], ["recovery", "research"])
        with self.assertRaises(WorkerJobError):
            WorkerJob("bad", "scope", WorkerJobKind.RESEARCH, {"url": "https://example.test"}, NOW + timedelta(minutes=1))


if __name__ == "__main__":
    unittest.main()
