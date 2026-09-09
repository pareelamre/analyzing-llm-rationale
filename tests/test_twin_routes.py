from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from analyzing_llm_rationale.twin.account_store import InMemoryAccountSnapshotStore
from analyzing_llm_rationale.twin.mandates import InMemoryMandateStore, Mandate, approve
from analyzing_llm_rationale.twin.models import AccountScope, CommandState, TradeIntent
from analyzing_llm_rationale.twin.operator import InMemoryPauseStore, OperatorService
from analyzing_llm_rationale.twin.routes import create_operator_router
from analyzing_llm_rationale.twin.store import InMemoryTwinStore
from analyzing_llm_rationale.twin.strategy import InMemoryStrategyStore, StrategyCycle
from analyzing_llm_rationale.twin.worker import InMemoryWorkerJobs

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def scope(scope_id: str, owner_id: str) -> AccountScope:
    return AccountScope(
        scope_id, owner_id, "kalshi", "private-account-ref", "shadow", "USD",
        "private-connection-ref", 1, NOW - timedelta(days=2),
    )


def intent(scope_id: str, suffix: str) -> TradeIntent:
    return TradeIntent(
        f"intent-{suffix}", scope_id, 1, f"kalshi:demo:KX{suffix}", "BUY_YES",
        Decimal("1"), Decimal("0.40"), "IOC", f"forecast-{suffix}", None,
        "policy-v1", "foresea_edge_v1", f"market-{suffix}",
        Decimal("0.01"), Decimal("0.01"), NOW + timedelta(hours=1), NOW,
    )


def mandate(owner_id: str, scope_id: str) -> Mandate:
    draft = Mandate(
        "mandate-001", owner_id, scope_id, "foresea_edge_v1", NOW + timedelta(days=1),
        account_epoch=1, venue="kalshi", max_capital="10", max_loss="5",
        max_model_usd="1", max_model_tokens=1000, max_model_requests=5,
        model_hash="a" * 64, config_hash="b" * 64, readiness_hash="c" * 64,
        release_hash="d" * 64, identity_hash="e" * 64, created_at=NOW,
    )
    return approve(draft, owner_id=owner_id, expected_hash=draft.digest(), now=NOW)


class TwinOperatorRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = "owner-001"
        self.twin = InMemoryTwinStore()
        self.twin.register_account(
            scope("scope-001", self.owner), venue_available_cash=Decimal("20"),
            loss_limit=Decimal("10"),
        )
        self.twin.register_account(
            scope("scope-other", "owner-other"), venue_available_cash=Decimal("99"),
            loss_limit=Decimal("99"),
        )
        for suffix in ("001", "002", "003"):
            trade = intent("scope-001", suffix)
            self.twin.reserve_intent(trade, cash=Decimal("1"), max_loss=Decimal("1"), now=NOW)
        first = self.twin.command_for_intent(intent("scope-001", "001"))
        claim = self.twin.claim_command(first.id, worker_id="worker-001", now=NOW)
        self.twin.transition_command(
            first.id, target=CommandState.ACKNOWLEDGED,
            fence=claim.fence, worker_id=claim.worker_id,
        )

        self.strategies = InMemoryStrategyStore()
        for index in range(3):
            self.strategies.record_cycle(StrategyCycle(
                f"cycle-{index}", "PASS", f"reason-{index}", created_at=NOW - timedelta(minutes=index),
                account_scope_id="scope-001",
            ))
        mandates = InMemoryMandateStore()
        mandates.create(mandate(self.owner, "scope-001"))
        self.jobs = InMemoryWorkerJobs()
        service = OperatorService(
            self.twin, InMemoryAccountSnapshotStore(), self.strategies, mandates,
            InMemoryPauseStore(), self.jobs,
            readiness=lambda _owner, account: {
                "status": "collecting", "release_hash": "d" * 64,
                "account_epoch": account.account_epoch,
            },
            clock=lambda: NOW,
        )
        self.service = service
        app = FastAPI()
        app.include_router(create_operator_router(
            service, resolve_owner=lambda request: request.headers.get("x-owner", ""),
        ))
        self.client = TestClient(app)
        self.headers = {"x-owner": self.owner}

    def test_status_portfolio_and_readiness_are_owner_scoped_and_secret_free(self):
        status = self.client.get("/twin/status", headers=self.headers)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["active_mandate_count"], 1)
        mandate_view = status.json()["mandates"][0]
        self.assertFalse(mandate_view["live"])
        self.assertEqual(mandate_view["max_capital"], "10")
        self.assertEqual(mandate_view["limits"]["max_model_tokens"], 1000)
        self.assertEqual(status.json()["stale_accounts"], 1)
        self.assertFalse(status.json()["new_exposure_allowed"])

        portfolio = self.client.get("/twin/portfolio", headers=self.headers).json()
        self.assertEqual(len(portfolio["accounts"]), 1)
        account = portfolio["accounts"][0]
        self.assertEqual(account["snapshot_status"], "missing")
        self.assertNotIn("connection_ref", account)
        self.assertNotIn("venue_account_ref", account)
        readiness = self.client.get("/twin/readiness", headers=self.headers).json()
        self.assertEqual(readiness["accounts"][0]["status"], "collecting")

    def test_decision_and_command_lists_have_bounded_stable_pagination(self):
        first = self.client.get("/twin/decisions?limit=2", headers=self.headers).json()
        self.assertEqual([item["key"] for item in first["items"]], ["cycle-0", "cycle-1"])
        self.assertIsNotNone(first["next_cursor"])
        second = self.client.get(
            "/twin/decisions", params={"limit": 2, "cursor": first["next_cursor"]},
            headers=self.headers,
        ).json()
        self.assertEqual([item["key"] for item in second["items"]], ["cycle-2"])
        commands = self.client.get("/twin/commands?limit=2", headers=self.headers).json()
        self.assertEqual(len(commands["items"]), 2)
        self.assertNotIn("worker_id", str(commands))

    def test_pause_and_cancel_are_idempotent_durable_requests(self):
        body = {"paused": True, "reason": "owner_kill_switch", "idempotency_key": "pause-001"}
        first = self.client.post("/twin/pause", json=body, headers=self.headers)
        replay = self.client.post("/twin/pause", json=body, headers=self.headers)
        self.assertEqual(first.json(), replay.json())
        conflict = self.client.post(
            "/twin/pause", json={**body, "paused": False}, headers=self.headers,
        )
        self.assertEqual(conflict.status_code, 409)

        command_id = self.twin.command_for_intent(intent("scope-001", "001")).id
        request = {"idempotency_key": "cancel-001"}
        queued = self.client.post(
            f"/twin/commands/{command_id}/cancel", json=request, headers=self.headers,
        )
        self.service.clock = lambda: NOW + timedelta(hours=3)
        repeated = self.client.post(
            f"/twin/commands/{command_id}/cancel", json=request, headers=self.headers,
        )
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(queued.json(), repeated.json())
        self.assertEqual(len(self.jobs.due(now=NOW)), 1)

    def test_cross_owner_and_premature_cancel_fail_closed(self):
        command_id = self.twin.command_for_intent(intent("scope-001", "001")).id
        self.assertEqual(self.client.post(
            f"/twin/commands/{command_id}/cancel", json={"idempotency_key": "cancel-002"},
            headers={"x-owner": "owner-other"},
        ).status_code, 404)
        reserved = self.twin.command_for_intent(intent("scope-001", "002"))
        self.assertEqual(self.client.post(
            f"/twin/commands/{reserved.id}/cancel", json={"idempotency_key": "cancel-003"},
            headers=self.headers,
        ).status_code, 409)
        self.assertEqual(self.client.get("/twin/status").status_code, 403)


if __name__ == "__main__":
    unittest.main()
