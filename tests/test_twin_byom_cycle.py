"""Offline acceptance of the shipped BYOM factory, SQL accounting and workers.

Only market/evidence/model inputs and calibration history are synthetic. The
factory, research parser, budget, finalizer, continuation, risk, execution and
durable stores are the application implementations. No real performance claim.
"""
import json
import os
import shutil
import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import yaml

from analyzing_llm_rationale import datastore_sql
from analyzing_llm_rationale.providers import ChatProvider
from analyzing_llm_rationale.twin import runtime_app
from analyzing_llm_rationale.twin.account_store import DatastoreAccountSnapshotStore
from analyzing_llm_rationale.twin.cycle_runtime import (
    DatastoreStrategyRunStore,
    InMemoryStrategyRunStore,
    StrategyRun,
)
from analyzing_llm_rationale.twin.research_gateway import (
    SYSTEM_PROMPT,
    _hash,
    load_research_runtime_policy,
)
from analyzing_llm_rationale.twin.scheduler import ensure_shadow_cycle_job
from analyzing_llm_rationale.twin.strategy import DatastoreStrategyStore
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    InMemoryWorkerJobs,
    TwinResearchWorker,
    TwinWorker,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
    WorkerJobStatus,
)
from tests.test_twin_market_capture import Gateway
from tests.test_twin_strategy import calibration_rows

ROOT = Path(__file__).resolve().parents[1]


class SyntheticModel(ChatProvider):
    model_name = "synthetic-byom"
    request_timeout_s = 60

    def __init__(self):
        self.calls = 0

    def chat_completion_with_usage(self, messages, temperature, max_tokens):
        self.calls += 1
        inputs = json.loads(messages[-1]["content"])
        forecast = {
            "p_yes": .75, "uncertainty_low": .65, "uncertainty_high": .85,
            "uncertainty_provenance": "Synthetic acceptance fixture; no measured skill.",
            "supporting_evidence_ids": [inputs["public_evidence"][0]["id"]],
            "contrary_evidence_ids": [],
            "expires_at": (datetime.fromisoformat(inputs["as_of"]) + timedelta(hours=1)).isoformat(),
        }
        return {"response": json.dumps(forecast), "usage": {
            "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        }}


class QueuedDispatcher:
    def __init__(self):
        self.ids = []

    def enqueue(self, job):
        self.ids.append(job.id)
        return "synthetic-task-" + job.id


class NoExternalEvidence:
    def search(self, _query, *, limit):
        return ()


@contextmanager
def synthetic_endpoint(model):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions" or self.headers.get("Authorization") != "Bearer fixture-key":
                self.send_error(401)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = model.chat_completion_with_usage(body["messages"], body["temperature"], body["max_tokens"])
            encoded = json.dumps({
                "model": model.model_name,
                "choices": [{"message": {"role": "assistant", "content": result["response"]}}],
                "usage": result["usage"],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class ByomCycleAcceptanceTests(unittest.TestCase):
    def test_concurrent_sql_continuations_keep_one_original_schedule(self):
        with TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"FORESEA_DATASTORE_BACKEND": "sql"},
        ):
            clients = [datastore_sql.Client(path=str(Path(directory) / "state.sqlite3")) for _ in range(2)]
            try:
                now = datetime.now(timezone.utc)
                stores = [DatastoreStrategyRunStore(client) for client in clients]
                run = stores[0].create(StrategyRun("synthetic-cycle", "shadow-synthetic", 1, "fixture-v1", now))
                jobs = [DatastoreWorkerJobs(client) for client in clients]
                payload = {field: "synthetic-id" for field in (
                    "research_assignment_id", "budget_reservation_id", "market_snapshot_id",
                    "evidence_set_id", "model_config_id", "budget_key_id",
                )}
                payload["strategy_cycle_id"] = run.id
                research = WorkerJob("synthetic-research", run.account_scope_id, WorkerJobKind.RESEARCH,
                                     payload, now + timedelta(minutes=5), created_at=now)
                ready = threading.Barrier(2)

                def stage(index):
                    # SQL uses BEGIN IMMEDIATE: writers cannot both read an
                    # absent row inside separate transactions. Synchronize
                    # before transaction entry, rather than deadlocking reads.
                    ready.wait(timeout=10)
                    return runtime_app._stage_strategy_continuation(
                        jobs[index], stores[index], research, now=now + timedelta(seconds=index),
                    )

                with ThreadPoolExecutor(max_workers=2) as callers:
                    first, second = callers.map(stage, range(2))
                self.assertEqual(first, second)
                self.assertIn(first.deadline, (now + timedelta(minutes=10), now + timedelta(minutes=10, seconds=1)))
                self.assertEqual(len(list(clients[0].query(kind="TwinWorkerJob").fetch())), 1)
            finally:
                for client in clients:
                    client.close()

    def test_continuation_retry_preserves_first_deadline_and_rejects_scope_collision(self):
        now = datetime.now(timezone.utc)
        runs = InMemoryStrategyRunStore()
        run = runs.create(StrategyRun("synthetic-cycle", "shadow-synthetic", 1, "fixture-v1", now))
        jobs = InMemoryWorkerJobs()
        payload = {field: "synthetic-id" for field in (
            "research_assignment_id", "budget_reservation_id", "market_snapshot_id",
            "evidence_set_id", "model_config_id", "budget_key_id",
        )}
        payload["strategy_cycle_id"] = run.id
        research = WorkerJob("synthetic-research", run.account_scope_id, WorkerJobKind.RESEARCH,
                             payload, now + timedelta(minutes=5), created_at=now)
        first = runtime_app._stage_strategy_continuation(jobs, runs, research, now=now)
        repeated = runtime_app._stage_strategy_continuation(jobs, runs, research,
                                                           now=now + timedelta(seconds=30))
        self.assertEqual(repeated, first)
        self.assertEqual(repeated.deadline, now + timedelta(minutes=10))
        for change in (
            {"account_scope_id": "shadow-other-scope"},
            {"kind": WorkerJobKind.RECOVERY},
            {"payload": {**first.payload, "config_release_id": "other-release"}},
            {"not_before": now + timedelta(minutes=1)},
        ):
            with self.subTest(change=change):
                colliding = InMemoryWorkerJobs()
                original = colliding.add(replace(first, **change))
                with self.assertRaises(WorkerJobError):
                    runtime_app._stage_strategy_continuation(colliding, runs, research,
                                                            now=now + timedelta(seconds=30))
                self.assertEqual(colliding.get(first.id), original)
        with self.assertRaisesRegex(WorkerJobError, "schema is unsupported"):
            replace(first, schema_version=2)

        worker = TwinWorker(jobs, worker_id="late-delivery", reconcile_startup=lambda: True)
        self.assertTrue(worker.start())
        try:
            expired = worker.handle(first.id, now=now + timedelta(minutes=11),
                                    maintain=lambda _job: self.fail("Expired continuation executed"))
            self.assertEqual(expired, {"status": "expired"})
            self.assertEqual(jobs.get(first.id).status, WorkerJobStatus.EXPIRED)
            self.assertEqual(jobs.get(first.id).deadline, first.deadline)
        finally:
            worker.shutdown()

    def test_factory_cycle_research_execution_accounting_and_receipt_recovery(self):
        for venue in ("kalshi", "polymarket"):
            with self.subTest(venue=venue), TemporaryDirectory() as directory:
                model = SyntheticModel()
                with synthetic_endpoint(model) as endpoint:
                    self._cycle(Path(directory), venue, model, endpoint)

    def _cycle(self, directory, venue, model, endpoint):
        config = directory / "config"
        config.mkdir()
        for name in ("twin.yaml", "models.yaml"):
            shutil.copyfile(ROOT / "deploy" / "byom" / name, config / name)
        policy_data = yaml.safe_load((config / "twin.yaml").read_text(encoding="utf-8"))
        policy_data["research"]["price_valid_until"] = (
            datetime.now(timezone.utc) + timedelta(days=1)
        ).isoformat()
        (config / "twin.yaml").write_text(yaml.safe_dump(policy_data), encoding="utf-8")
        models = yaml.safe_load((config / "models.yaml").read_text(encoding="utf-8"))
        for field in ("local_model_name", "router_model_name"):
            models["models"]["byom-local"][field] = "synthetic-byom"
        models["models"]["byom-local"]["api_base_url"] = endpoint
        (config / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
        policy = load_research_runtime_policy(config / "twin.yaml", config / "models.yaml")
        observations = tuple({**row,
            "model_hash": _hash({"provider": policy.model.provider_id, "model": policy.model.model_id}),
            "prompt_hash": _hash(SYSTEM_PROMPT),
        } for row in calibration_rows())
        gateway = Gateway()
        discover = gateway.discover
        gateway.discover = lambda selected, **kwargs: discover(selected, **kwargs) if selected == venue else []
        fetch = gateway.fetch

        def titled(selected, identifier):
            market, books = fetch(selected, identifier)
            return {**market, "title": "SYNTHETIC BYOM acceptance outcome"}, books

        gateway.fetch = titled
        dispatcher = QueuedDispatcher()
        environment = {
            "FORESEA_DATASTORE_BACKEND": "sql",
            "FORESEA_DATASTORE_PATH": str(directory / "twin.sqlite3"),
            "FORESEA_TWIN_WORKER_ROLE": "maintenance",
            "FORESEA_TWIN_MODE": "shadow", "FORESEA_TWIN_LIVE_CAPITAL": "0",
            "FORESEA_TWIN_LIVE_MANDATE": "", "FORESEA_TWIN_CONFIG_DIR": str(config),
            "FORESEA_TWIN_MAINTENANCE_DISPATCH_SECRET": "a" * 64,
            "FORESEA_TWIN_RESEARCH_DISPATCH_SECRET": "b" * 64,
            "FORESEA_TWIN_RESEARCH_GATEWAY_SECRET": "c" * 64,
            "FORESEA_TWIN_MAINTENANCE_URL": "http://127.0.0.1:8081",
            "FORESEA_TWIN_RESEARCH_URL": "http://127.0.0.1:8082",
            "FORESEA_TWIN_SCHEDULER_ACCOUNTS": "local-scheduler",
            "FORESEA_TWIN_DISPATCHER_ACCOUNTS": "local-dispatcher",
            "FORESEA_TWIN_RESEARCH_ACCOUNTS": "local-research",
        }
        connect, getaddrinfo = socket.socket.connect, socket.getaddrinfo

        def loopback_connect(sock, address):
            if address[0] != "127.0.0.1":
                raise AssertionError("Offline acceptance forbids external connections")
            return connect(sock, address)

        def loopback_dns(host, *args, **kwargs):
            if host != "127.0.0.1":
                raise AssertionError("Offline acceptance forbids external DNS")
            return getaddrinfo(host, *args, **kwargs)

        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(runtime_app, "_build_dispatcher", return_value=dispatcher),
            mock.patch.object(runtime_app, "LiveMarketDataGateway", return_value=gateway),
            mock.patch.object(runtime_app, "NewsPipelinePublicArticleGateway", return_value=NoExternalEvidence()),
            mock.patch.object(runtime_app, "_calibration_observations", return_value=observations),
            mock.patch.object(runtime_app, "create_private_worker_app", side_effect=lambda runtime: runtime),
            mock.patch.object(runtime_app, "init_observability"),
            mock.patch("socket.socket.connect", loopback_connect),
            mock.patch("socket.getaddrinfo", loopback_dns),
        ):
            with mock.patch.dict(os.environ, {"FORESEA_TWIN_BYOM_API_KEY": "fixture-key"}):
                provider = policy.build_provider()
            self.addCleanup(provider._session.close)
            runtime = runtime_app.create_environment_app()
            self.assertTrue(runtime.start())
            try:
                job = ensure_shadow_cycle_job(runtime.jobs, runtime.cycle_schedule,
                                              now=datetime.now(timezone.utc))
                first = runtime.maintenance_worker.handle(job.id, now=datetime.now(timezone.utc),
                                                          maintain=runtime.maintenance_operation)
                self.assertEqual(first["status"], "pending", first)
                research_id = dispatcher.ids[-1]
                # The accepted result is durable before the finalizer executes.
                with mock.patch.object(runtime.research_gateway, "_finalize_result",
                                       side_effect=TimeoutError("synthetic finalizer interruption")):
                    with self.assertRaisesRegex(TimeoutError, "synthetic finalizer"):
                        TwinResearchWorker(runtime.research_gateway, worker_id="before-restart").handle(
                            research_id, now=datetime.now(timezone.utc),
                            research=lambda assignment: runtime_app._research_operation(
                                assignment, policy, runtime.research_gateway, provider),
                        )
                self.assertEqual(model.calls, 1)
            finally:
                runtime.shutdown()
                runtime.jobs._client.close()
            # New factory + new SQL clients, with only a saved completion.
            runtime = runtime_app.create_environment_app()
            try:
                recovered = TwinResearchWorker(runtime.research_gateway, worker_id="after-restart").handle(
                    research_id, now=datetime.now(timezone.utc) + timedelta(seconds=126),
                    research=lambda _assignment: self.fail("Recovery repeated model inference"),
                )
                self.assertEqual(recovered["status"], "completed", recovered)
                self.assertTrue(runtime.start())
                continuation = dispatcher.ids[-1]
                completed = runtime.maintenance_worker.handle(
                    continuation, now=datetime.now(timezone.utc), maintain=runtime.maintenance_operation,
                )
                self.assertEqual(completed["decision"], "INTENT", completed)
                self.assertEqual(completed["run_phase"], "complete", completed)
                client = runtime.jobs._client
                run = DatastoreStrategyRunStore(client).get(job.payload["strategy_cycle_id"])
                cycle = DatastoreStrategyStore(client).get_cycle(run.id)
                snapshots = DatastoreAccountSnapshotStore(client)
                account = snapshots.load(job.account_scope_id)
                self.assertEqual(len(account.orders), 1)
                self.assertGreater(len(account.fills), 0)
                self.assertEqual(account.reserved_cash, Decimal("0"))
                basis = sum((holding.basis for holding in account.holdings), Decimal("0"))
                self.assertGreater(basis, 0)
                self.assertGreater(account.fees_paid, 0)
                # Simulator basis includes entry fees; fees_paid is also
                # reported separately and must not be counted twice.
                self.assertEqual(account.available_cash + basis, Decimal("1000"))
                self.assertEqual(account.fees_paid, sum(
                    (Decimal(str(fill["fee"])) for fill in account.fills), Decimal("0"),
                ))
                duplicate = runtime.maintenance_worker.handle(
                    continuation, now=datetime.now(timezone.utc), maintain=runtime.maintenance_operation,
                )
                self.assertEqual(duplicate, completed)
                self.assertEqual(snapshots.load(job.account_scope_id), account)
                self.assertEqual(model.calls, 1)
                self.assertEqual(cycle.intent.account_scope_id, account.scope_id)
            finally:
                runtime.shutdown()
                runtime.jobs._client.close()


if __name__ == "__main__":
    unittest.main()
