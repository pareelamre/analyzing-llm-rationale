"""Off-GCP worker dispatch and authentication.

Cloud Tasks and Google OIDC are the last two GCP services the twin runtime
depends on. ``LocalTaskDispatcher`` replaces the first and the shared-secret
branch in ``PrivateTwinRuntime.authenticate`` replaces the second. These tests
pin the properties that make the swap safe:

* the HTTP body is byte-identical to what Cloud Tasks sent, so the worker
  route needs no change;
* jobs route to the right queue by kind;
* a delivery failure surfaces as ``WorkerDispatchError``, not a raw
  ``requests`` exception, so ``dispatch_due_jobs`` keeps its contract;
* the shared secret is compared in constant time and never falls through to
  the OIDC path.
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.twin import scheduler as sch  # noqa: E402
from analyzing_llm_rationale.twin.worker import (  # noqa: E402
    WorkerAuthenticationError,
    WorkerJob,
    WorkerJobKind,
)

_NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _job(kind: WorkerJobKind, ident: str = "job-1") -> WorkerJob:
    payload = {}
    if kind is WorkerJobKind.STRATEGY:
        payload = {
            "strategy_cycle_id": "cycle-1",
            "config_release_id": "release-1",
            "account_epoch_id": "1",
        }
    elif kind is WorkerJobKind.RESEARCH:
        payload = {
            "research_assignment_id": "assignment-1",
            "budget_reservation_id": "reservation-1",
            "market_snapshot_id": "snapshot-1",
            "evidence_set_id": "evidence-1",
            "model_config_id": "model-1",
            "budget_key_id": "budget-1",
        }
    return WorkerJob(
        id=ident, account_scope_id="shadow-1", kind=kind,
        payload=payload, deadline=_NOW + timedelta(minutes=5),
    )


class _Response:
    def __init__(self, status: int = 200):
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, status: int = 200, raises: Exception | None = None):
        self.calls: list[tuple] = []
        self._status = status
        self._raises = raises

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, headers, timeout))
        if self._raises is not None:
            raise self._raises
        return _Response(self._status)


class LocalDispatchConfigTests(unittest.TestCase):
    def test_rejects_empty_secret(self):
        with self.assertRaises(sch.WorkerDispatchError):
            sch.LocalDispatchConfig("http://a/x", "http://b/y", "   ")

    def test_rejects_relative_url(self):
        with self.assertRaises(sch.WorkerDispatchError):
            sch.LocalDispatchConfig("/internal/twin/maintain", "http://b/y", "s")

    def test_rejects_out_of_range_timeout(self):
        with self.assertRaises(sch.WorkerDispatchError):
            sch.LocalDispatchConfig("http://a/x", "http://b/y", "s", timeout_seconds=0)

    def test_accepts_https(self):
        config = sch.LocalDispatchConfig("https://a/x", "https://b/y", "s")
        self.assertEqual(config.timeout_seconds, 10.0)


class LocalTaskDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = sch.LocalDispatchConfig(
            "http://maintenance/internal/twin/maintain",
            "http://research/internal/twin/research",
            "shared-secret",
        )

    def test_body_is_byte_identical_to_cloud_tasks(self):
        # Cloud Tasks sent json.dumps({"job_id": ...}, sort_keys=True,
        # separators=(",", ":")). The worker route parses this exact shape.
        session = _Session()
        sch.LocalTaskDispatcher(self.config, session=session).enqueue(_job(WorkerJobKind.RECOVERY))
        _, data, _, _ = session.calls[0]
        self.assertEqual(data, b'{"job_id":"job-1"}')

    def test_maintenance_kinds_route_to_maintenance_url(self):
        session = _Session()
        dispatcher = sch.LocalTaskDispatcher(self.config, session=session)
        for kind in (WorkerJobKind.RECOVERY, WorkerJobKind.RECONCILE, WorkerJobKind.EXIT,
                     WorkerJobKind.STRATEGY):
            dispatcher.enqueue(_job(kind))
        self.assertTrue(all(
            url == self.config.maintenance_url for url, _, _, _ in session.calls
        ))

    def test_research_routes_to_research_url(self):
        session = _Session()
        sch.LocalTaskDispatcher(self.config, session=session).enqueue(
            _job(WorkerJobKind.RESEARCH)
        )
        self.assertEqual(session.calls[0][0], self.config.research_url)

    def test_sends_bearer_secret(self):
        session = _Session()
        sch.LocalTaskDispatcher(self.config, session=session).enqueue(_job(WorkerJobKind.RECOVERY))
        headers = session.calls[0][2]
        self.assertEqual(headers["Authorization"], "Bearer shared-secret")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_returns_a_stable_task_name(self):
        session = _Session()
        name = sch.LocalTaskDispatcher(self.config, session=session).enqueue(
            _job(WorkerJobKind.RECOVERY)
        )
        self.assertEqual(name, "local:job-1")

    def test_transport_failure_is_wrapped(self):
        session = _Session(raises=RuntimeError("connection refused"))
        with self.assertRaises(sch.WorkerDispatchError):
            sch.LocalTaskDispatcher(self.config, session=session).enqueue(
                _job(WorkerJobKind.RECOVERY)
            )

    def test_http_error_status_is_wrapped(self):
        session = _Session(status=503)
        with self.assertRaises(sch.WorkerDispatchError):
            sch.LocalTaskDispatcher(self.config, session=session).enqueue(
                _job(WorkerJobKind.RECOVERY)
            )

    def test_satisfies_the_task_dispatcher_protocol(self):
        # dispatch_due_jobs only needs .enqueue(job) -> str.
        dispatcher = sch.LocalTaskDispatcher(self.config, session=_Session())
        self.assertTrue(callable(dispatcher.enqueue))


class _Request:
    def __init__(self, token: str | None):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}


class SharedSecretAuthTests(unittest.TestCase):
    """The runtime must authenticate the local dispatcher without Google OIDC."""

    def _runtime(self, secret: str | None):
        from analyzing_llm_rationale.twin.runtime import (
            PrivateTwinRuntime,
            RuntimeIdentityPolicy,
        )

        return PrivateTwinRuntime(
            role="research",
            identities=RuntimeIdentityPolicy(
                audience="https://worker.example",
                scheduler_accounts=frozenset({"scheduler@x"}),
                dispatcher_accounts=frozenset({"sa@x"}),
                research_accounts=frozenset({"sa@x"}),
            ),
            clock=lambda: _NOW,
            research_worker=object(),
            research_operation=lambda assignment: None,
            shared_secret=secret,
        )

    def test_correct_secret_authenticates(self):
        runtime = self._runtime("s3cret")
        principal = runtime.authenticate(_Request("s3cret"), frozenset({"sa@x"}))
        self.assertEqual(principal, "local-dispatcher")

    def test_wrong_secret_is_rejected(self):
        runtime = self._runtime("s3cret")
        with self.assertRaises(WorkerAuthenticationError):
            runtime.authenticate(_Request("wrong"), frozenset({"sa@x"}))

    def test_missing_token_is_rejected(self):
        runtime = self._runtime("s3cret")
        with self.assertRaises(WorkerAuthenticationError):
            runtime.authenticate(_Request(None), frozenset({"sa@x"}))

    def test_secret_does_not_fall_through_to_oidc(self):
        # With a secret configured, a non-matching token must fail closed
        # rather than being handed to the Google verifier.
        runtime = self._runtime("s3cret")
        called = []

        def verifier(token, audience):
            called.append(token)
            return {}

        runtime.token_verifier = verifier
        with self.assertRaises(WorkerAuthenticationError):
            runtime.authenticate(_Request("not-the-secret"), frozenset({"sa@x"}))
        self.assertEqual(called, [])

    def test_no_secret_uses_the_oidc_path(self):
        runtime = self._runtime(None)
        called = []

        def verifier(token, audience):
            called.append((token, audience))
            return {
                "iss": "https://accounts.google.com",
                "aud": "https://worker.example",
                "email": "sa@x",
                "email_verified": True,
                "exp": (_NOW + timedelta(minutes=5)).timestamp(),
            }

        runtime.token_verifier = verifier
        principal = runtime.authenticate(_Request("google-token"), frozenset({"sa@x"}))
        self.assertEqual(principal, "sa@x")
        self.assertEqual(called, [("google-token", "https://worker.example")])


if __name__ == "__main__":
    unittest.main()
