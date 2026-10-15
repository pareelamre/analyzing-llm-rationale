import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from google.api_core.exceptions import AlreadyExists

from analyzing_llm_rationale.twin.scheduler import (
    CloudTasksConfig,
    CloudTasksDispatcher,
    ShadowCycleSchedule,
    dispatch_due_jobs,
    ensure_shadow_cycle_job,
)
from analyzing_llm_rationale.twin.strategy import strategy_cycle_key_for_identity
from analyzing_llm_rationale.twin.worker import (
    InMemoryWorkerJobs,
    MaintenanceResearchJobGateway,
    ResearchAssignment,
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
            "budget_key_id": "foresea-edge:scope-001:2025-01-01",
        "strategy_cycle_id": "strategy-cycle-001",
        }
    elif kind is WorkerJobKind.STRATEGY:
        payload = {
            "strategy_cycle_id": "strategy-cycle-001",
            "config_release_id": "foresea-edge-shadow-v1",
            "account_epoch_id": "1",
        }
    return WorkerJob(job_id, "scope-001", kind, payload, deadline)


class TwinWorkerTests(unittest.TestCase):
    def test_legacy_research_payload_remains_readable_after_cycle_link_migration(self):
        payload = dict(job("research", kind=WorkerJobKind.RESEARCH).payload)
        payload.pop("strategy_cycle_id")
        legacy = WorkerJob(
            "legacy-research", "scope-001", WorkerJobKind.RESEARCH, payload,
            NOW + timedelta(minutes=1), worker_id="research-worker",
        )
        assignment = ResearchAssignment.from_job(legacy)
        self.assertTrue(assignment.strategy_cycle_id.startswith("legacy-research-cycle:"))

    def test_shadow_cycle_production_is_idempotent_per_release_bucket(self):
        jobs = InMemoryWorkerJobs()
        schedule = ShadowCycleSchedule(
            "shadow-scope:foresea-edge-v1", "foresea-edge-shadow-v1",
            account_epoch=7,
        )
        first = ensure_shadow_cycle_job(jobs, schedule, now=NOW + timedelta(seconds=1))
        duplicate = ensure_shadow_cycle_job(jobs, schedule, now=NOW + timedelta(seconds=299))
        self.assertEqual(len(jobs.due(now=NOW + timedelta(seconds=299))), 1)
        following = ensure_shadow_cycle_job(jobs, schedule, now=NOW + timedelta(seconds=300))

        self.assertEqual(first.id, duplicate.id)
        self.assertNotEqual(first.id, following.id)
        self.assertEqual(first.kind, WorkerJobKind.STRATEGY)
        self.assertEqual(first.payload["strategy_cycle_id"], duplicate.payload["strategy_cycle_id"])
        self.assertEqual(first.payload["account_epoch_id"], "7")
        self.assertEqual(first.payload["strategy_cycle_id"], strategy_cycle_key_for_identity(
            scope_id=schedule.account_scope_id, account_epoch=schedule.account_epoch,
            now=NOW + timedelta(seconds=1), config_version=schedule.config_release_id,
            bucket_seconds=schedule.bucket_seconds,
        ))

    def test_shadow_cycle_schedule_rejects_live_scope_and_oversized_deadline(self):
        with self.assertRaisesRegex(Exception, "shadow account scope"):
            ShadowCycleSchedule("live-scope", "release-v1")
        with self.assertRaisesRegex(Exception, "fit inside"):
            ShadowCycleSchedule(
                "shadow-scope", "release-v1", bucket_seconds=300,
                deadline_seconds=301,
            )

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

    def test_stale_recovery_expires_deadlines_and_refences_lost_leases(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("expired", deadline=NOW + timedelta(seconds=1)))
        jobs.add(job("leased", deadline=NOW + timedelta(minutes=1)))
        jobs.claim("leased", worker_id="lost-worker", now=NOW, lease_seconds=1)

        recovered = jobs.recover_stale(now=NOW + timedelta(seconds=2))

        self.assertEqual([item.id for item in recovered], ["expired", "leased"])
        expired = jobs.get("expired")
        self.assertEqual(expired.status, WorkerJobStatus.EXPIRED)
        self.assertEqual(expired.completed_result, {"status": "expired"})
        leased = jobs.get("leased")
        self.assertEqual(leased.status, WorkerJobStatus.QUEUED)
        self.assertIsNone(leased.worker_id)
        self.assertIsNone(leased.lease_expires_at)
        reclaimed = jobs.claim(
            "leased", worker_id="replacement", now=NOW + timedelta(seconds=2),
        )
        self.assertEqual(reclaimed.fence, 2)

    def test_maintenance_lease_can_cover_the_platform_request_deadline(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job(deadline=NOW + timedelta(minutes=5)))
        observed = []
        worker = TwinWorker(
            jobs, worker_id="worker", reconcile_startup=lambda: True,
            lease_seconds=125,
        )
        worker.start()

        worker.handle(
            "job-001", now=NOW,
            maintain=lambda claimed: observed.append(claimed.lease_expires_at) or {
                "status": "complete",
            },
        )

        self.assertEqual(observed, [NOW + timedelta(seconds=125)])

    def test_startup_dependency_failure_keeps_health_process_alive_but_unready(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job())
        worker = TwinWorker(
            jobs,
            worker_id="new",
            reconcile_startup=lambda: (_ for _ in ()).throw(PermissionError("datastore unavailable")),
        )

        with self.assertLogs("analyzing_llm_rationale.twin.worker", level="ERROR"):
            self.assertFalse(worker.start())

        self.assertFalse(worker.execution_ready)
        self.assertFalse(worker.accepting_work)
        self.assertEqual(
            worker.handle("job-001", now=NOW, maintain=lambda _: {"status": "unsafe"}),
            {"status": "draining"},
        )

    def test_readiness_can_recover_after_durable_reconciliation_finishes(self):
        jobs = InMemoryWorkerJobs()
        reconciled = [False]
        worker = TwinWorker(
            jobs, worker_id="new", reconcile_startup=lambda: reconciled[0],
        )

        self.assertFalse(worker.start())
        self.assertTrue(worker.accepting_work)
        reconciled[0] = True

        self.assertTrue(worker.refresh_readiness())
        self.assertTrue(worker.execution_ready)
        self.assertTrue(worker.accepting_work)

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

    def test_research_result_transport_is_finalized_before_job_completion(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        finalized = []
        continuations = []

        def finalize(assignment, completion):
            finalized.append((assignment.fence, completion.result_payload))
            return ResearchCompletion(
                "completed", research_result_id="result-derived",
                usage_record_id="usage-derived",
            )

        worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                jobs, authorize_assignment=lambda _: None,
                finalize_result=finalize,
                after_complete=lambda assignment, completed: continuations.append(
                    (assignment.strategy_cycle_id, completed.status.value)
                ),
            ),
            worker_id="research-worker",
        )
        result = worker.handle(
            "research", now=NOW,
            research=lambda _: ResearchCompletion(
                "completed", result_payload={"schema_version": 1},
                actual_usd="0", actual_tokens=12,
            ),
        )

        self.assertEqual(finalized, [(1, {"schema_version": 1})])
        self.assertEqual(continuations, [("strategy-cycle-001", "completed")])
        self.assertEqual(result["research_result_id"], "result-derived")
        self.assertNotIn("result_payload", jobs.get("research").completed_result)

    def test_research_validation_failure_is_persisted_as_degraded(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        finalized = []

        def finalize(_assignment, completion):
            finalized.append((completion.status, completion.reason))
            if completion.status == "completed":
                raise WorkerDegraded("research_result_identity_mismatch")
            return completion

        worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                jobs, authorize_assignment=lambda _: None,
                finalize_result=finalize,
            ),
            worker_id="research-worker",
        )
        result = worker.handle(
            "research", now=NOW,
            research=lambda _: ResearchCompletion(
                "completed", result_payload={"schema_version": 1},
                actual_usd="0", actual_tokens=12,
            ),
        )

        self.assertEqual(
            finalized,
            [("completed", None), ("degraded", "research_result_identity_mismatch")],
        )
        self.assertEqual(
            result,
            {"status": "degraded", "reason": "research_result_identity_mismatch"},
        )
        self.assertEqual(jobs.get("research").status, WorkerJobStatus.DEGRADED)

    def test_saved_research_completion_recovers_after_finalizer_failure(self):
        from analyzing_llm_rationale import datastore_sql
        from analyzing_llm_rationale.twin.worker import (
            DatastoreResearchCompletionStore,
            DatastoreWorkerJobs,
        )

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"},
        ):
            path = str(Path(tmp) / "jobs.sqlite3")
            client = datastore_sql.Client(path=path)
            try:
                jobs = DatastoreWorkerJobs(client)
                jobs.add(job(
                    "research", kind=WorkerJobKind.RESEARCH,
                    deadline=NOW + timedelta(minutes=5),
                ))
                stored = DatastoreResearchCompletionStore(client)
                finalizations = []
                model_calls = []

                def finalize(_assignment, completion):
                    finalizations.append((
                        completion.result_payload, completion.actual_usd,
                        completion.actual_tokens,
                    ))
                    if len(finalizations) == 1:
                        raise TimeoutError("injected failure after durable receipt")
                    return ResearchCompletion(
                        "completed", research_result_id="result-001",
                        usage_record_id="usage-001",
                    )

                def research(_assignment):
                    model_calls.append(1)
                    return ResearchCompletion(
                        "completed", result_payload={"schema_version": 1},
                        actual_usd="0", actual_tokens=12,
                    )

                gateway = MaintenanceResearchJobGateway(
                    jobs, authorize_assignment=lambda _assignment: None,
                    finalize_result=finalize, completion_store=stored,
                )
                worker = TwinResearchWorker(gateway, worker_id="research-worker")
                with self.assertRaisesRegex(TimeoutError, "durable receipt"):
                    worker.handle("research", now=NOW, research=research)
                self.assertEqual(jobs.get("research").status, WorkerJobStatus.RUNNING)
                saved_assignment = ResearchAssignment.from_job(jobs.get("research"))
                self.assertEqual(stored.get("research").actual_tokens, 12)
                with self.assertRaisesRegex(WorkerDegraded, "too_large"):
                    DatastoreResearchCompletionStore(
                        client, max_payload_bytes=1,
                    ).record(saved_assignment, ResearchCompletion(
                        "completed", result_payload={"schema_version": 1},
                        actual_usd="0", actual_tokens=12,
                    ))
                with self.assertRaisesRegex(WorkerJobError, "conflicts"):
                    stored.record(
                        saved_assignment, ResearchCompletion(
                            "completed", result_payload={"schema_version": 1},
                            actual_usd="0", actual_tokens=13,
                        ),
                    )

                client.close()
                client = datastore_sql.Client(path=path)
                jobs = DatastoreWorkerJobs(client)

                recovered = MaintenanceResearchJobGateway(
                    jobs,
                    authorize_assignment=lambda _assignment: self.fail(
                        "a saved completion must not claim another model budget"
                    ),
                    finalize_result=finalize,
                    completion_store=DatastoreResearchCompletionStore(client),
                )
                replay = TwinResearchWorker(recovered, worker_id="restart-worker")
                result = replay.handle(
                    "research", now=NOW + timedelta(seconds=126), research=research,
                )
                self.assertEqual(result["research_result_id"], "result-001")
                self.assertEqual(len(model_calls), 1)
                self.assertEqual(finalizations, [
                    ({"schema_version": 1}, "0", 12),
                    ({"schema_version": 1}, "0", 12),
                ])
                self.assertEqual(jobs.get("research").status, WorkerJobStatus.COMPLETED)
                self.assertIsNone(
                    DatastoreResearchCompletionStore(client).get("research"),
                )
            finally:
                client.close()

    def test_stale_research_completion_does_not_run_finalizer(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        first = jobs.claim("research", worker_id="old-worker", now=NOW)
        jobs.claim("research", worker_id="new-worker", now=NOW + timedelta(seconds=31))
        finalized = []
        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=lambda _: None,
            finalize_result=lambda _assignment, result: finalized.append(1) or result,
        )
        with self.assertRaisesRegex(WorkerJobError, "stale assignment"):
            gateway.complete(
                ResearchAssignment.from_job(first),
                ResearchCompletion(
                    "completed", research_result_id="result-001",
                    usage_record_id="usage-001",
                ),
                now=NOW + timedelta(seconds=31),
            )
        self.assertEqual(finalized, [])

    def test_research_lease_outlasts_private_request_timeout(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job(
            "research", kind=WorkerJobKind.RESEARCH,
            deadline=NOW + timedelta(minutes=5),
        ))
        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=lambda _: None,
        )
        self.assertIsNotNone(gateway.claim(
            "research", worker_id="first", now=NOW,
        ))
        self.assertIsNone(gateway.claim(
            "research", worker_id="second", now=NOW + timedelta(seconds=31),
        ))
        self.assertIsNotNone(gateway.claim(
            "research", worker_id="third", now=NOW + timedelta(seconds=126),
        ))

    def test_corrupt_saved_completion_degrades_without_another_model_call(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        degraded = []

        class CorruptCompletionStore:
            def get(self, _job_id):
                raise WorkerJobError("durable research completion is malformed")

            def record(self, _assignment, _completion):
                raise AssertionError("corrupt receipt must not be overwritten")

            def discard(self, _job_id):
                return None

        def finalize(_assignment, completion):
            degraded.append((completion.status, completion.reason))
            return completion

        gateway = MaintenanceResearchJobGateway(
            jobs,
            authorize_assignment=lambda _: self.fail("must not claim budget"),
            finalize_result=finalize,
            completion_store=CorruptCompletionStore(),
        )
        worker = TwinResearchWorker(gateway, worker_id="research-worker")
        result = worker.handle(
            "research", now=NOW,
            research=lambda _: self.fail("must not call model"),
        )
        self.assertEqual(result, {
            "status": "degraded", "reason": "research_completion_unreadable",
        })
        self.assertEqual(degraded, [("degraded", "research_completion_unreadable")])
        self.assertEqual(jobs.get("research").status, WorkerJobStatus.DEGRADED)

    def test_corrupt_durable_completion_is_quarantined_after_degradation(self):
        from analyzing_llm_rationale import datastore_sql
        from analyzing_llm_rationale.twin.worker import (
            DatastoreResearchCompletionStore,
            DatastoreWorkerJobs,
        )

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"},
        ):
            client = datastore_sql.Client(path=str(Path(tmp) / "jobs.sqlite3"))
            try:
                jobs = DatastoreWorkerJobs(client)
                jobs.add(job(
                    "research", kind=WorkerJobKind.RESEARCH,
                    deadline=NOW + timedelta(minutes=5),
                ))
                first = jobs.claim(
                    "research", worker_id="first", now=NOW, lease_seconds=125,
                )
                stored = DatastoreResearchCompletionStore(client)
                stored.record(
                    ResearchAssignment.from_job(first),
                    ResearchCompletion(
                        "completed", result_payload={"schema_version": 1},
                        actual_usd="0", actual_tokens=12,
                    ),
                )
                entity = client.get(stored._key("research"))
                entity["payload_sha256"] = "corrupted"
                client.put(entity)
                gateway = MaintenanceResearchJobGateway(
                    jobs, authorize_assignment=lambda _: self.fail(
                        "corrupt receipt must not claim model budget"
                    ),
                    finalize_result=lambda _assignment, completion: completion,
                    completion_store=stored,
                )
                replay = TwinResearchWorker(gateway, worker_id="second")
                result = replay.handle(
                    "research", now=NOW + timedelta(seconds=126),
                    research=lambda _: self.fail("must not call model"),
                )
                self.assertEqual(result["reason"], "research_completion_unreadable")
                self.assertEqual(jobs.get("research").status, WorkerJobStatus.DEGRADED)
                self.assertIsNone(stored.get("research"))
            finally:
                client.close()

    def test_oversized_completion_degrades_after_one_model_call(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        model_calls = []
        finalized = []

        class OversizeCompletionStore:
            def get(self, _job_id):
                return None

            def record(self, _assignment, _completion):
                raise WorkerDegraded("research_completion_too_large")

            def discard(self, _job_id):
                return None

        def finalize(_assignment, completion):
            finalized.append(completion.status)
            return completion

        worker = TwinResearchWorker(
            MaintenanceResearchJobGateway(
                jobs, authorize_assignment=lambda _: None,
                finalize_result=finalize,
                completion_store=OversizeCompletionStore(),
            ),
            worker_id="research-worker",
        )
        result = worker.handle(
            "research", now=NOW,
            research=lambda _: model_calls.append(1) or ResearchCompletion(
                "completed", result_payload={"schema_version": 1},
                actual_usd="0", actual_tokens=12,
            ),
        )
        self.assertEqual(result, {
            "status": "degraded", "reason": "research_completion_too_large",
        })
        self.assertEqual(len(model_calls), 1)
        self.assertEqual(finalized, ["degraded"])

    def test_completed_research_retries_lost_continuation_dispatch(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(job("research", kind=WorkerJobKind.RESEARCH))
        dispatches = []

        def after_complete(_assignment, _job):
            dispatches.append(1)
            if len(dispatches) == 1:
                raise TimeoutError("lost continuation acknowledgement")

        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=lambda _: None,
            after_complete=after_complete,
        )
        worker = TwinResearchWorker(gateway, worker_id="research-worker")
        def research(_assignment):
            return ResearchCompletion(
                "completed", research_result_id="result-001",
                usage_record_id="usage-001",
            )
        with self.assertRaisesRegex(TimeoutError, "continuation acknowledgement"):
            worker.handle("research", now=NOW, research=research)
        self.assertEqual(jobs.get("research").status, WorkerJobStatus.COMPLETED)
        self.assertEqual(
            worker.handle("research", now=NOW + timedelta(seconds=1), research=research),
            {"status": "completed", "research_result_id": "result-001",
             "usage_record_id": "usage-001"},
        )
        self.assertEqual(len(dispatches), 2)

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
