"""Durable delayed delivery without changing the identity of legacy jobs."""

import json
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.scheduler import (
    CloudTasksConfig,
    CloudTasksDispatcher,
    LocalDispatchConfig,
    LocalTaskDispatcher,
    dispatch_due_jobs,
)
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    InMemoryWorkerJobs,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
)
from tests.test_twin_recovery import NOW


class ScheduledWorkerJobsTests(unittest.TestCase):
    def task(self):
        return WorkerJob(
            "delayed-recovery",
            "scope",
            WorkerJobKind.RECOVERY,
            {"command_id": "command"},
            NOW + timedelta(minutes=5),
            created_at=NOW,
            not_before=NOW + timedelta(seconds=30),
        )

    def test_delayed_jobs_cannot_be_claimed_or_dispatched_early_after_restart(self):
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FORESEA_DATASTORE_BACKEND": "sql"}),
        ):
            path = str(Path(root) / "jobs.sqlite3")
            with closing(Client(path=path)) as db:
                DatastoreWorkerJobs(db).add(self.task())
            with closing(Client(path=path)) as db:
                for jobs in (InMemoryWorkerJobs(), DatastoreWorkerJobs(db)):
                    with self.subTest(backend=type(jobs).__name__):
                        jobs.add(self.task())
                        dispatcher = Mock()
                        self.assertIsNone(jobs.claim(self.task().id, worker_id="early", now=NOW))
                        self.assertEqual(dispatch_due_jobs(jobs, dispatcher, now=NOW), ())
                        dispatcher.enqueue.assert_not_called()
                        due = self.task().not_before
                        self.assertEqual(jobs.due(now=due), (jobs.get(self.task().id),))
                        claimed = jobs.claim(self.task().id, worker_id="due", now=due)
                        self.assertEqual(claimed.fence, 1)

    def test_legacy_job_identity_is_unchanged_and_delayed_identity_is_bound(self):
        legacy = WorkerJob(
            "old-job", "scope", WorkerJobKind.RECOVERY, {}, NOW + timedelta(minutes=5)
        )
        old = {
            "id": legacy.id,
            "account_scope_id": legacy.account_scope_id,
            "kind": legacy.kind.value,
            "payload": {},
            "deadline": legacy.deadline.isoformat(),
            "schema_version": 1,
        }
        digest = sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(DatastoreWorkerJobs._identity(legacy), digest)
        delayed = replace(legacy, not_before=NOW + timedelta(seconds=30))
        self.assertNotEqual(DatastoreWorkerJobs._identity(delayed), digest)
        self.assertNotEqual(
            DatastoreWorkerJobs._identity(delayed),
            DatastoreWorkerJobs._identity(replace(delayed, not_before=NOW + timedelta(seconds=31))),
        )

    def test_invalid_delays_are_rejected(self):
        for value in (
            NOW.replace(tzinfo=None),
            NOW - timedelta(seconds=1),
            NOW + timedelta(minutes=5),
        ):
            with self.subTest(value=value), self.assertRaises(WorkerJobError):
                replace(self.task(), not_before=value)

    def test_cloud_tasks_receives_the_delivery_time_and_id_only_body(self):
        config = CloudTasksConfig(
            "project",
            "us-central1",
            "maintenance",
            "research",
            "https://maintenance.example/internal/twin/maintain",
            "https://research.example/internal/twin/research",
            "dispatcher@project.iam.gserviceaccount.com",
            "https://maintenance.example",
            "https://research.example",
        )
        client = Mock()
        client.queue_path.return_value = "projects/project/locations/us-central1/queues/maintenance"
        client.create_task.return_value = SimpleNamespace(name="created")
        CloudTasksDispatcher(config, client).enqueue(self.task())
        task = client.create_task.call_args.kwargs["request"]["task"]
        self.assertEqual(
            task["schedule_time"].ToDatetime(tzinfo=NOW.tzinfo), self.task().not_before
        )
        self.assertEqual(json.loads(task["http_request"]["body"]), {"job_id": self.task().id})

    def test_local_dispatch_waits_for_due_scan_instead_of_sleeping_or_posting_early(self):
        created = datetime.now(timezone.utc)
        task = replace(
            self.task(),
            created_at=created,
            deadline=created + timedelta(minutes=5),
            not_before=created + timedelta(seconds=30),
        )
        session = Mock()
        session.post.return_value.status_code = 200
        dispatcher = LocalTaskDispatcher(
            LocalDispatchConfig(
                "http://127.0.0.1:8081/internal/twin/maintain",
                "http://127.0.0.1:8082/internal/twin/research",
                "fixture-maintenance",
                "fixture-research",
            ),
            session,
        )
        self.assertEqual(dispatcher.enqueue(task), task.id)
        session.post.assert_not_called()
        with patch("analyzing_llm_rationale.twin.scheduler.datetime") as clock:
            clock.now.return_value = task.not_before
            dispatcher.enqueue(task)
        session.post.assert_called_once()
