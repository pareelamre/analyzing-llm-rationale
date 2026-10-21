"""Real private maintenance completion preserves an explicit operator pause."""

import unittest
from datetime import timedelta
from decimal import Decimal
from unittest.mock import Mock

from analyzing_llm_rationale.twin.runtime_app import _maintenance_operation
from analyzing_llm_rationale.twin.worker import (
    TwinWorker,
    WorkerDegraded,
    WorkerJob,
    WorkerJobKind,
    WorkerJobStatus,
)
from tests import test_twin_native_lifecycle as native


class NativeRecoveryPauseTests(unittest.TestCase):
    fixture_type = native.NativeLifecycleTests

    def setUp(self):
        self.case = self.fixture_type(methodName="runTest")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.run_count = 0

    def run_job(self, adapter, *, kind=WorkerJobKind.RECOVERY):
        self.run_count += 1
        queued = WorkerJob(
            "pause-boundary-" + str(self.run_count),
            self.case.scope.id,
            kind,
            {"command_id": self.case.command.id, "account_epoch_id": "1"},
            native.NOW + timedelta(minutes=5),
            created_at=native.NOW,
        )
        self.case.jobs.add(queued)
        worker = TwinWorker(self.case.jobs, worker_id="boundary", reconcile_startup=lambda: True)
        worker.start()
        def maintain(job):
            return _maintenance_operation(
                self.case.jobs, None, job, native_lifecycle=adapter, account_maintenance=adapter
            )
        result = worker.handle(queued.id, now=self.case.time, maintain=maintain)
        return queued, result

    def assert_paused_and_replayable(self, adapter):
        queued, result = self.run_job(adapter)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(self.case.jobs.get(queued.id).status, WorkerJobStatus.PAUSED)
        self.assertEqual(
            self.case.store.projection(self.case.scope.id).reserved_cash, Decimal("1")
        )
        restarted = TwinWorker(
            self.case.jobs, worker_id="restarted", reconcile_startup=lambda: True
        )
        restarted.start()
        duplicate = restarted.handle(
            queued.id,
            now=self.case.time + timedelta(seconds=1),
            maintain=lambda _: self.fail("paused native work must not run twice"),
        )
        self.assertEqual(duplicate, result)
        return result

    def test_unconfigured_native_adapter_persists_a_pause(self):
        self.assert_paused_and_replayable(None)

    def test_native_degradation_pauses_without_exporting_private_reason(self):
        adapter = Mock(side_effect=WorkerDegraded("fixture_private_reason"))
        result = self.assert_paused_and_replayable(adapter)
        self.assertNotIn("fixture_private_reason", str(result))
        adapter.assert_called_once()

    def test_exhausted_native_followup_budget_persists_a_pause(self):
        self.assert_paused_and_replayable(
            Mock(
                return_value={
                    "status": "reconciliation_required",
                    "followup_budget_exhausted": True,
                }
            )
        )

    def test_malformed_native_result_cannot_falsely_complete(self):
        for value in (None, {}, {"status": []}, {"status": "unexpected"}):
            with self.subTest(value=value):
                self.assert_paused_and_replayable(Mock(return_value=value))

    def test_pending_native_work_keeps_its_automatic_continuation(self):
        expected = {"status": "pending", "followup_job_id": "saved-followup"}
        queued, result = self.run_job(Mock(return_value=expected))
        self.assertEqual(result, expected)
        self.assertEqual(self.case.jobs.get(queued.id).status, WorkerJobStatus.COMPLETED)

    def test_complete_native_work_still_completes(self):
        queued, result = self.run_job(Mock(return_value={"status": "complete"}))
        self.assertEqual(result, {"status": "complete"})
        self.assertEqual(self.case.jobs.get(queued.id).status, WorkerJobStatus.COMPLETED)

    def test_non_native_degradation_keeps_its_existing_status(self):
        queued, result = self.run_job(
            Mock(side_effect=WorkerDegraded("data_unavailable")), kind=WorkerJobKind.RECONCILE
        )
        self.assertEqual(result, {"status": "degraded", "reason": "data_unavailable"})
        self.assertEqual(self.case.jobs.get(queued.id).status, WorkerJobStatus.DEGRADED)
