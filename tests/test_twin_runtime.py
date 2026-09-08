import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from fastapi.testclient import TestClient

from analyzing_llm_rationale.twin.runtime import (
    HttpResearchJobGateway,
    PrivateTwinRuntime,
    RuntimeIdentityPolicy,
    create_private_worker_app,
)
from analyzing_llm_rationale.twin.runtime_app import _assert_shadow_only
from analyzing_llm_rationale.twin.worker import (
    InMemoryWorkerJobs,
    MaintenanceResearchJobGateway,
    ResearchCompletion,
    TwinResearchWorker,
    TwinWorker,
    WorkerJob,
    WorkerJobKind,
    WorkerRole,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
AUDIENCE = "https://twin-maintenance.example"
SCHEDULER = "scheduler@example.iam.gserviceaccount.com"
DISPATCHER = "dispatcher@example.iam.gserviceaccount.com"
RESEARCH = "research@example.iam.gserviceaccount.com"


def verifier(token, audience):
    identities = {
        "scheduler-token": SCHEDULER,
        "dispatcher-token": DISPATCHER,
        "research-token": RESEARCH,
        "intruder-token": "intruder@example.iam.gserviceaccount.com",
    }
    return {
        "iss": "https://accounts.google.com",
        "aud": audience,
        "email": identities[token],
        "email_verified": True,
        "exp": (NOW + timedelta(minutes=5)).timestamp(),
    }


def research_job(job_id="research-job"):
    return WorkerJob(
        job_id, "scope-001", WorkerJobKind.RESEARCH,
        {
            "research_assignment_id": "assignment-001",
            "budget_reservation_id": "budget-001",
            "market_snapshot_id": "snapshot-001",
            "evidence_set_id": "evidence-001",
            "model_config_id": "model-001",
            "budget_key_id": "foresea-edge:scope-001:2025-01-01",
        },
        NOW + timedelta(minutes=1),
    )


class Dispatcher:
    def __init__(self):
        self.ids = []

    def enqueue(self, job):
        self.ids.append(job.id)
        return f"task/{job.id}"


class PrivateTwinRuntimeTests(unittest.TestCase):
    def maintenance_runtime(self, *, startup=True):
        jobs = InMemoryWorkerJobs()
        jobs.add(WorkerJob(
            "maintenance-job", "scope-001", WorkerJobKind.RECONCILE,
            {"account_snapshot_id": "snapshot-001"}, NOW + timedelta(minutes=1),
        ))
        jobs.add(research_job())
        authorized = []
        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=lambda item: authorized.append(item.budget_reservation_id),
        )
        runtime = PrivateTwinRuntime(
            WorkerRole.MAINTENANCE,
            RuntimeIdentityPolicy(
                AUDIENCE, frozenset({SCHEDULER}), frozenset({DISPATCHER}),
                frozenset({RESEARCH}),
            ),
            lambda: NOW,
            jobs=jobs,
            dispatcher=Dispatcher(),
            maintenance_worker=TwinWorker(
                jobs, worker_id="maintenance-worker", reconcile_startup=lambda: startup,
            ),
            maintenance_operation=lambda _: {"status": "complete"},
            research_gateway=gateway,
            token_verifier=verifier,
        )
        return runtime, authorized

    @staticmethod
    def auth(token):
        return {"Authorization": f"Bearer {token}"}

    def test_maintenance_surface_is_private_role_scoped_and_ready_after_recovery(self):
        runtime, _ = self.maintenance_runtime()
        with TestClient(create_private_worker_app(runtime)) as client:
            self.assertEqual(client.get("/health").json()["role"], "maintenance")
            self.assertEqual(client.get("/ready").status_code, 200)
            self.assertEqual(client.get("/trading/orders").status_code, 404)
            response = client.post(
                "/internal/twin/maintain", json={"job_id": "maintenance-job"},
                headers=self.auth("dispatcher-token"),
            )
            self.assertEqual(response.json(), {"status": "complete"})

    def test_environment_runtime_rejects_live_capital_or_authority(self):
        base = {
            "FORESEA_TWIN_MODE": "shadow",
            "FORESEA_TWIN_LIVE_CAPITAL": "0",
            "FORESEA_TWIN_LIVE_MANDATE": "",
        }
        with mock.patch.dict("os.environ", base, clear=True):
            _assert_shadow_only()
        for override in (
            {"FORESEA_TWIN_MODE": "live"},
            {"FORESEA_TWIN_LIVE_CAPITAL": "1"},
            {"FORESEA_TWIN_LIVE_MANDATE": "mandate-001"},
        ):
            with self.subTest(override=override), mock.patch.dict(
                "os.environ", {**base, **override}, clear=True,
            ), self.assertRaises(RuntimeError):
                _assert_shadow_only()

    def test_startup_reconciliation_failure_keeps_readiness_closed(self):
        runtime, _ = self.maintenance_runtime(startup=False)
        with TestClient(create_private_worker_app(runtime)) as client:
            self.assertEqual(client.get("/ready").status_code, 503)

    def test_route_identity_cannot_be_replaced_by_spoofed_queue_headers(self):
        runtime, _ = self.maintenance_runtime()
        with TestClient(create_private_worker_app(runtime)) as client:
            anonymous = client.post(
                "/internal/twin/maintain", json={"job_id": "maintenance-job"},
                headers={"X-CloudTasks-QueueName": "twin-maintenance"},
            )
            self.assertEqual(anonymous.status_code, 401)
            unauthorized = client.post(
                "/internal/twin/maintain", json={"job_id": "maintenance-job"},
                headers=self.auth("scheduler-token"),
            )
            self.assertEqual(unauthorized.status_code, 401)

    def test_scheduler_dispatch_and_budgeted_research_claim_use_distinct_identities(self):
        runtime, authorized = self.maintenance_runtime()
        with TestClient(create_private_worker_app(runtime)) as client:
            dispatch = client.post(
                "/internal/twin/dispatch", headers=self.auth("scheduler-token"),
            )
            self.assertEqual(dispatch.status_code, 200)
            self.assertEqual(dispatch.json()["tasks_enqueued"], 2)
            claim = client.post(
                "/internal/twin/research-jobs/research-job/claim",
                headers=self.auth("research-token"),
            )
            self.assertEqual(claim.status_code, 200)
            assignment = claim.json()["assignment"]
            self.assertEqual(authorized, ["budget-001"])
            result = client.post(
                "/internal/twin/research-jobs/research-job/result",
                headers=self.auth("research-token"),
                json={
                    "fence": assignment["fence"], "status": "completed",
                    "research_result_id": "result-001", "usage_record_id": "usage-001",
                },
            )
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["research_result_id"], "result-001")

    def test_research_surface_cannot_reach_maintenance_or_public_routes(self):
        jobs = InMemoryWorkerJobs()
        jobs.add(research_job())
        gateway = MaintenanceResearchJobGateway(
            jobs, authorize_assignment=lambda _: None,
        )
        runtime = PrivateTwinRuntime(
            WorkerRole.RESEARCH,
            RuntimeIdentityPolicy(
                "https://twin-research.example", frozenset(),
                frozenset({DISPATCHER}), frozenset(),
            ),
            lambda: NOW,
            research_worker=TwinResearchWorker(gateway, worker_id="research-worker"),
            research_operation=lambda _: ResearchCompletion(
                "completed", research_result_id="result-001", usage_record_id="usage-001",
            ),
            token_verifier=verifier,
        )
        with TestClient(create_private_worker_app(runtime)) as client:
            response = client.post(
                "/internal/twin/research", json={"job_id": "research-job"},
                headers=self.auth("dispatcher-token"),
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "completed")
            self.assertEqual(client.post("/internal/twin/maintain").status_code, 404)
            self.assertEqual(client.get("/trading/orders").status_code, 404)

    def test_http_research_gateway_sends_only_identity_bound_contracts(self):
        assignment = {
            "job_id": "research-job", "worker_id": "research-worker", "fence": 2,
            "deadline": (NOW + timedelta(minutes=1)).isoformat(),
            "research_assignment_id": "assignment-001",
            "budget_reservation_id": "budget-001",
            "market_snapshot_id": "snapshot-001", "evidence_set_id": "evidence-001",
            "model_config_id": "model-001",
            "budget_key_id": "foresea-edge:scope-001:2025-01-01",
        }

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.calls = []

            def request(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                if url.endswith("/claim"):
                    return Response({"status": "claimed", "assignment": assignment})
                if url.endswith("/result"):
                    return Response({"status": "completed", "research_result_id": "result-001"})
                return Response({"status": "running", "completed_result": None})

        session = Session()
        gateway = HttpResearchJobGateway(
            "https://maintenance.example", audience=AUDIENCE, session=session,
            token_fetcher=lambda audience: f"token-for-{audience}",
        )
        claimed = gateway.claim("research-job", worker_id="ignored", now=NOW)
        self.assertEqual(claimed.fence, 2)
        completed = gateway.complete(
            claimed, ResearchCompletion(
                "completed", research_result_id="result-001", usage_record_id="usage-001",
            ), now=NOW,
        )
        self.assertEqual(completed["research_result_id"], "result-001")
        _, _, call = session.calls[-1]
        self.assertEqual(call["headers"], {"Authorization": f"Bearer token-for-{AUDIENCE}"})
        self.assertEqual(
            set(call["json"]),
            {"fence", "status", "research_result_id", "usage_record_id"},
        )


if __name__ == "__main__":
    unittest.main()
