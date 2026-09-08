import json
import unittest
from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import AlreadyExists

from analyzing_llm_rationale.twin.scheduler import (
    CloudTasksConfig,
    CloudTasksDispatcher,
    dispatch_due_jobs,
)
from analyzing_llm_rationale.twin.worker import (
    InMemoryWorkerJobs,
    MaintenanceResearchJobGateway,
    ResearchCompletion,
    TwinResearchWorker,
    TwinWorker,
    WorkerAuthenticationError,
    WorkerDegraded,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
    WorkerPaused,
    WorkerRole,
    bounded_safe_read,
    require_worker_request,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def job(job_id="job-001", *, kind=WorkerJobKind.RECONCILE, deadline=NOW + timedelta(minutes=1)):
    payload = {"market_id": "market-001"}
    if kind is WorkerJobKind.RESEARCH:
        payload = {
            "research_assignment_id": "assignment-001",
            "budget_reservation_id": "budget-001",
            "market_snapshot_id": "snapshot-001",
            "evidence_set_id": "evidence-001",
            "model_config_id": "model-001",
        }
    return WorkerJob(job_id, "scope-001", kind, payload, deadline)


class TwinWorkerTests(unittest.TestCase):
    def test_private_worker_rejects_public_request(self):
        require_worker_request("valid", expected_token="valid")
        with self.assertRaises(WorkerAuthenticationError):
            require_worker_request(None, expected_token="valid")

    def test_duplicate_delivery_returns_saved_result_and_never_runs_twice(self):
        jobs, calls = InMemoryWorkerJobs(), []
        jobs.add(job())
        worker = TwinWorker(jobs, worker_id="worker-a", reconcile_startup=lambda: True)
        worker.start()
        first = worker.handle("job-001", now=NOW, maintain=lambda _: calls.append("maintenance") or {"ok": True})
        restarted = TwinWorker(jobs, worker_id="worker-b", reconcile_startup=lambda: True)
        restarted.start()
        second = restarted.handle("job-001", now=NOW, maintain=lambda _: calls.append("again") or {})
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
        deadline_jobs = InMemoryWorkerJobs()
        deadline_jobs.add(job(deadline=NOW + timedelta(seconds=3)))
        deadline_claim = deadline_jobs.claim(
            "job-001", worker_id="bounded", now=NOW, lease_seconds=30,
        )
        self.assertEqual(deadline_claim.lease_expires_at, NOW + timedelta(seconds=3))
        worker = TwinWorker(jobs, worker_id="new", reconcile_startup=lambda: False)
        self.assertFalse(worker.start())

    def test_priority_and_payload_constraints_keep_maintenance_ahead_of_research(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        jobs.add(job("recovery", kind=WorkerJobKind.RECOVERY))
        self.assertEqual([item.id for item in jobs.due(now=NOW)], ["recovery", "research"])
        with self.assertRaises(WorkerJobError):
            WorkerJob("bad", "scope", WorkerJobKind.RESEARCH, {"url": "https://example.test"}, NOW + timedelta(minutes=1))

    def test_stale_fence_cannot_complete_even_when_worker_id_is_reused(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job())
        first = jobs.claim("job-001", worker_id="same-worker", now=NOW, lease_seconds=1)
        second = jobs.claim(
            "job-001", worker_id="same-worker", now=NOW + timedelta(seconds=2),
        )
        with self.assertRaisesRegex(WorkerJobError, "stale"):
            jobs.complete(
                "job-001", worker_id="same-worker", fence=first.fence,
                result={"ok": True}, now=NOW + timedelta(seconds=2),
            )
        completed = jobs.complete(
            "job-001", worker_id="same-worker", fence=second.fence,
            result={"ok": True}, now=NOW + timedelta(seconds=2),
        )
        self.assertEqual(completed.status, WorkerJobStatus.COMPLETED)

    def test_shutdown_and_deadline_stop_new_work(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job())
        worker = TwinWorker(jobs, worker_id="worker", reconcile_startup=lambda: True)
        worker.start()
        worker.shutdown()
        self.assertEqual(
            worker.handle("job-001", now=NOW, maintain=lambda _: {}),
            {"status": "draining"},
        )
        expired_jobs = InMemoryWorkerJobs()
        expired_jobs.add(job(deadline=NOW))
        expired = TwinWorker(expired_jobs, worker_id="worker", reconcile_startup=lambda: True)
        expired.start()
        self.assertEqual(
            expired.handle("job-001", now=NOW, maintain=lambda _: {}),
            {"status": "expired"},
        )

    def test_stale_detection_and_hard_ambiguity_pause_are_durable(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job())
        jobs.claim("job-001", worker_id="dead", now=NOW, lease_seconds=1)
        self.assertEqual(
            [item.id for item in jobs.stale(now=NOW + timedelta(seconds=2))],
            ["job-001"],
        )
        worker = TwinWorker(jobs, worker_id="replacement", reconcile_startup=lambda: True)
        worker.start()
        result = worker.handle(
            "job-001", now=NOW + timedelta(seconds=2),
            maintain=lambda _: (_ for _ in ()).throw(WorkerPaused("submission_unknown")),
        )
        self.assertEqual(result, {"status": "paused", "reason": "submission_unknown"})
        self.assertEqual(jobs.get("job-001").status, WorkerJobStatus.PAUSED)
        self.assertEqual(
            worker.handle(
                "job-001", now=NOW + timedelta(seconds=3),
                maintain=lambda _: self.fail("paused work must not run twice"),
            ),
            result,
        )

    def test_role_isolation_and_model_outage_do_not_block_maintenance(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        research_worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(jobs, authorize_assignment=lambda _: None),
            worker_id="research-worker",
        )
        result = research_worker.handle(
            "research", now=NOW,
            research=lambda _: (_ for _ in ()).throw(WorkerDegraded("model_unavailable")),
        )
        self.assertEqual(result, {"status": "degraded", "reason": "model_unavailable"})

        budget_jobs = InMemoryWorkerJobs()
        budget_jobs.add(job("budgeted", kind=WorkerJobKind.RESEARCH))
        budget_worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                budget_jobs,
                authorize_assignment=lambda _: (_ for _ in ()).throw(
                    WorkerDegraded("research_budget_exhausted")
                ),
            ), worker_id="research-worker",
        )
        budget_calls = []
        self.assertEqual(
            budget_worker.handle(
                "budgeted", now=NOW,
                research=lambda _: budget_calls.append(1) or ResearchCompletion(
                    "completed", research_result_id="result-001",
                    usage_record_id="usage-001",
                ),
            ),
            {"status": "degraded", "reason": "research_budget_exhausted"},
        )
        self.assertEqual(budget_calls, [])

        maintenance_jobs = InMemoryWorkerJobs()
        maintenance_jobs.add(job("maintenance"))
        calls = []
        maintenance = TwinWorker(
            maintenance_jobs, worker_id="maintenance-worker",
            reconcile_startup=lambda: True, role=WorkerRole.MAINTENANCE,
        )
        maintenance.start()
        self.assertEqual(
            maintenance.handle(
                "maintenance", now=NOW,
                maintain=lambda _: calls.append("reconciled") or {"status": "complete"},
            ),
            {"status": "complete"},
        )
        self.assertEqual(calls, ["reconciled"])
        with self.assertRaisesRegex(WorkerJobError, "narrow"):
            TwinWorker(
                jobs, worker_id="bad-research", reconcile_startup=lambda: True,
                role=WorkerRole.RESEARCH,
            )

    def test_research_gateway_accepts_only_typed_result_references(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(jobs, authorize_assignment=lambda _: None),
            worker_id="research-worker",
        )
        result = worker.handle(
            "research", now=NOW,
            research=lambda assignment: ResearchCompletion(
                "completed", research_result_id=f"result-{assignment.research_assignment_id}",
                usage_record_id="usage-001",
            ),
        )
        self.assertEqual(result["research_result_id"], "result-assignment-001")
        with self.assertRaises(WorkerJobError):
            ResearchCompletion("completed", research_result_id="result-without-usage")

    def test_safe_reads_retry_only_within_the_configured_budget(self):
        attempts, sleeps = [], []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("temporary")
            return "complete"

        self.assertEqual(
            bounded_safe_read(flaky, attempts=3, base_delay_seconds=0.5, sleep=sleeps.append),
            "complete",
        )
        self.assertEqual(sleeps, [0.5, 1.0])
        with self.assertRaises(WorkerDegraded):
            bounded_safe_read(
                lambda: (_ for _ in ()).throw(TimeoutError("down")),
                attempts=2, base_delay_seconds=0, sleep=lambda _: None,
            )

    def test_cloud_tasks_payload_is_id_only_and_duplicate_name_is_stable(self):
        class Response:
            name = "created"

        class FakeClient:
            def __init__(self):
                self.requests = []

            @staticmethod
            def queue_path(project, location, queue):
                return f"projects/{project}/locations/{location}/queues/{queue}"

            def create_task(self, *, request, timeout):
                self.requests.append((request, timeout))
                if len(self.requests) > 1:
                    raise AlreadyExists("duplicate")
                return Response()

        config = CloudTasksConfig(
            "project", "us-central1", "maintenance", "research",
            "https://maintenance.example/internal/twin/maintain",
            "https://research.example/internal/twin/research",
            "dispatcher@project.iam.gserviceaccount.com",
            "https://maintenance.example", "https://research.example",
        )
        client = FakeClient()
        dispatcher = CloudTasksDispatcher(config, client)
        task = job("stable-job")
        self.assertEqual(dispatcher.enqueue(task), "created")
        request, timeout = client.requests[0]
        self.assertEqual(json.loads(request["task"]["http_request"]["body"]), {"job_id": "stable-job"})
        self.assertNotIn("scope-001", request["task"]["http_request"]["body"].decode())
        self.assertEqual(timeout, 10.0)
        duplicate_name = dispatcher.enqueue(task)
        self.assertEqual(duplicate_name, request["task"]["name"])

    def test_dispatch_orders_maintenance_before_research(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        jobs.add(job("exit", kind=WorkerJobKind.EXIT))
        jobs.add(job("recovery", kind=WorkerJobKind.RECOVERY))

        class Dispatcher:
            def __init__(self):
                self.ids = []

            def enqueue(self, queued):
                self.ids.append(queued.id)
                return queued.id

        dispatcher = Dispatcher()
        self.assertEqual(
            dispatch_due_jobs(jobs, dispatcher, now=NOW),
            ("recovery", "exit", "research"),
        )


if __name__ == "__main__":
    unittest.main()
