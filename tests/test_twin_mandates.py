import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from analyzing_llm_rationale.twin.mandates import (
    InMemoryMandateStore,
    Mandate,
    MandateConflict,
    MandateError,
    PauseState,
    approve,
    authorize_mandate,
    revise,
    revoke,
)
from analyzing_llm_rationale.twin.models import AccountScope
from analyzing_llm_rationale.twin.routes import (
    MandateRuntime,
    MandateService,
    create_mandate_router,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def draft(**changes):
    values = {
        "id": "mandate-001", "owner_id": "owner-001", "account_scope_id": "scope-001",
        "strategy_version": "foresea_edge_v1", "expires_at": NOW + timedelta(days=1),
        "account_epoch": 2, "venue": "kalshi", "max_capital": "10", "max_loss": "5",
        "max_model_usd": "1", "max_model_tokens": 1000, "max_model_requests": 5,
        "model_hash": "a" * 64, "config_hash": "b" * 64, "readiness_hash": "c" * 64,
        "release_hash": "d" * 64, "identity_hash": "e" * 64, "created_at": NOW,
    }
    values.update(changes)
    return Mandate(**values)


class TwinMandateTests(unittest.TestCase):
    def test_owner_exact_hash_approval_expiry_and_idempotent_revocation(self):
        pending = draft()
        active = approve(pending, owner_id="owner-001", expected_hash=pending.digest(), now=NOW)
        self.assertTrue(active.active(now=NOW))
        self.assertFalse(active.active(now=active.expires_at))
        revoked = revoke(active, owner_id="owner-001", now=NOW)
        self.assertEqual(revoked, revoke(revoked, owner_id="owner-001", now=NOW + timedelta(hours=1)))
        with self.assertRaises(PermissionError):
            approve(pending, owner_id="other", expected_hash=pending.digest(), now=NOW)

    def test_stale_review_and_widened_budget_require_a_new_version(self):
        pending = draft()
        reviewed_hash = pending.digest()
        widened = replace(pending, max_capital="11")
        with self.assertRaisesRegex(MandateConflict, "changed after owner review"):
            approve(widened, owner_id="owner-001", expected_hash=reviewed_hash, now=NOW)
        active = approve(pending, owner_id="owner-001", expected_hash=reviewed_hash, now=NOW)
        revised = revise(active, max_capital="11")
        self.assertEqual(revised.version, 2)
        self.assertEqual(revised.parent_hash, active.digest())
        self.assertIsNone(revised.approved_hash)
        self.assertFalse(revised.active(now=NOW))

    def test_store_is_owner_scoped_versioned_and_replay_safe(self):
        store = InMemoryMandateStore()
        pending = store.create(draft())
        self.assertIsNone(store.get("other-owner", pending.id))
        active = approve(pending, owner_id=pending.owner_id, expected_hash=pending.digest(), now=NOW)
        saved = store.save_transition(pending, active, idempotency_key="approval-001")
        replay = store.save_transition(pending, replace(active, approved_at=NOW + timedelta(seconds=1)), idempotency_key="approval-001")
        self.assertEqual(replay, saved)
        version_two = revise(saved, max_loss="4")
        store.save_transition(saved, version_two, idempotency_key="revision-001")
        self.assertEqual([item.version for item in store.versions(pending.owner_id, pending.id)], [1, 2])

    def test_epoch_actions_and_every_pause_scope_are_enforced(self):
        active = approve(draft(), owner_id="owner-001", now=NOW)
        authorize_mandate(active, now=NOW, pause=PauseState(), account_epoch=2, action="BUY_YES")
        with self.assertRaisesRegex(MandateConflict, "epoch"):
            authorize_mandate(active, now=NOW, pause=PauseState(), account_epoch=3, action="BUY_YES")
        for pause in (
            PauseState(global_pause=True), PauseState(account_scopes=frozenset({"scope-001"})),
            PauseState(strategies=frozenset({"foresea_edge_v1"})), PauseState(venues=frozenset({"kalshi"})),
        ):
            with self.subTest(pause=pause), self.assertRaisesRegex(PermissionError, "paused"):
                authorize_mandate(active, now=NOW, pause=pause, account_epoch=2, action="BUY_YES")

    def test_live_draft_cannot_activate_from_t12_or_unbound_authority(self):
        live = draft(live=True)
        with self.assertRaisesRegex(PermissionError, "verified readiness"):
            approve(
                live, owner_id="owner-001", expected_hash=live.digest(), now=NOW,
                identity_hash=live.identity_hash, release_hash=live.release_hash,
                config_hash=live.config_hash, model_hash=live.model_hash,
                readiness_hash=live.readiness_hash,
            )

    def test_unknown_capabilities_and_storage_corruption_fail_closed(self):
        with self.assertRaisesRegex(MandateError, "bounded prediction-market"):
            draft(allowed_actions=("withdraw_funds",))
        stored = draft().to_storage()
        self.assertEqual(Mandate.from_storage(stored), draft())
        stored["max_capital"] = "NaN"
        with self.assertRaises(MandateError):
            Mandate.from_storage(stored)


class TwinMandateRouteTests(unittest.TestCase):
    def setUp(self):
        self.epoch = 2
        self.store = InMemoryMandateStore()

        def scope_resolver(scope_id):
            return AccountScope(
                scope_id, "owner-001", "kalshi", "account-001", "shadow", "USD",
                "connection-001", self.epoch, NOW - timedelta(days=1),
            )

        def runtime(_owner, _scope):
            return MandateRuntime("e" * 64, "d" * 64, "b" * 64, "a" * 64, "c" * 64, None)

        service = MandateService(self.store, resolve_scope=scope_resolver, resolve_runtime=runtime, clock=lambda: NOW)

        def owner(request):
            if request.headers.get("x-role") == "research":
                raise PermissionError("research role cannot manage mandates")
            return request.headers.get("x-owner", "")

        app = FastAPI()
        app.include_router(create_mandate_router(service, resolve_owner=owner))
        self.client = TestClient(app)
        self.body = {
            "client_request_id": "draft-request-001", "account_scope_id": "scope-001",
            "strategy_version": "foresea_edge_v1", "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "live": False, "allowed_actions": ["BUY_YES", "BUY_NO"], "max_capital": "10",
            "max_loss": "5", "max_model_usd": "1", "max_model_tokens": 1000,
            "max_model_requests": 5,
        }

    def test_owner_is_derived_and_cross_owner_reads_are_hidden(self):
        extra = {**self.body, "owner_id": "attacker"}
        self.assertEqual(self.client.post("/twin/mandates", json=extra, headers={"x-owner": "owner-001"}).status_code, 422)
        created = self.client.post("/twin/mandates", json=self.body, headers={"x-owner": "owner-001"})
        self.assertEqual(created.status_code, 200)
        self.assertEqual(len(created.json()["authority_hash"]), 64)
        mandate_id = created.json()["id"]
        self.assertEqual(self.client.get(f"/twin/mandates/{mandate_id}", headers={"x-owner": "other"}).status_code, 404)

    def test_exact_hash_approval_replay_and_epoch_change(self):
        created = self.client.post("/twin/mandates", json=self.body, headers={"x-owner": "owner-001"}).json()
        approval = {"expected_hash": created["authority_hash"], "idempotency_key": "approval-request-001"}
        first = self.client.post(f"/twin/mandates/{created['id']}/approve", json=approval, headers={"x-owner": "owner-001"})
        self.assertEqual(first.status_code, 200)
        replay = self.client.post(f"/twin/mandates/{created['id']}/approve", json=approval, headers={"x-owner": "owner-001"})
        self.assertEqual(replay.json(), first.json())

        other_body = {**self.body, "client_request_id": "draft-request-002"}
        other = self.client.post("/twin/mandates", json=other_body, headers={"x-owner": "owner-001"}).json()
        self.epoch = 3
        response = self.client.post(
            f"/twin/mandates/{other['id']}/approve",
            json={"expected_hash": other["authority_hash"], "idempotency_key": "approval-request-002"},
            headers={"x-owner": "owner-001"},
        )
        self.assertEqual(response.status_code, 409)

    def test_research_role_cannot_create_authority(self):
        response = self.client.post(
            "/twin/mandates", json=self.body,
            headers={"x-owner": "owner-001", "x-role": "research"},
        )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
