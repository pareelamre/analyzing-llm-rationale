import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import AccountScope, CommandState, InMemoryTwinStore, TradeIntent
from analyzing_llm_rationale.twin.execution import (
    ExecutionBlocked,
    ExecutionContext,
    SubmissionDisposition,
    SubmissionUnknown,
    submit_authorized_command,
    submit_claimed_command,
)
from analyzing_llm_rationale.twin.mandates import Mandate, approve, revoke
from analyzing_llm_rationale.twin.models import ProposalAction

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class DurableMemoryStore(InMemoryTwinStore):
    """Test stand-in for a live durable implementation."""

    durable = True


def scope(*, environment="shadow", epoch=1):
    return AccountScope(
        id=f"scope-{environment}-{epoch}", owner_id="owner-001", venue="kalshi",
        venue_account_ref="account-001", environment=environment, collateral_asset="USD",
        connection_ref="connection-001", account_epoch=epoch, created_at=NOW,
    )


def intent(active_scope):
    return TradeIntent(
        id="intent-001", account_scope_id=active_scope.id, account_epoch=active_scope.account_epoch,
        instrument_id=f"kalshi:{active_scope.environment}:KXTEST", action=ProposalAction.BUY_YES,
        quantity=Decimal("5"), limit_price=Decimal("0.5"), time_in_force="IOC",
        forecast_id="forecast-001", exit_reason=None, policy_version="policy-v1",
        strategy_version="strategy-v1", market_version="market-v1", fee_allowance=Decimal("0.12"),
        slippage_allowance=Decimal("0.08"), expires_at=NOW + timedelta(minutes=5), created_at=NOW,
    )


def reserved(*, environment="shadow", autonomous=True):
    active_scope = scope(environment=environment)
    store = DurableMemoryStore() if environment == "live" else InMemoryTwinStore()
    store.register_account(active_scope, venue_available_cash=Decimal("20"), loss_limit=Decimal("20"))
    trade_intent = intent(active_scope)
    store.reserve_intent(trade_intent, cash=Decimal("2.70"), max_loss=Decimal("2.70"), now=NOW)
    command = store.command_for_intent(trade_intent)
    claim = store.claim_command(command.id, worker_id="worker-001", now=NOW, lease_seconds=30)
    assert claim is not None
    mandate = approve(
        Mandate(
            "mandate-001", "owner-001", active_scope.id, "strategy-v1", NOW + timedelta(days=1),
            live=environment == "live", account_epoch=active_scope.account_epoch, venue="kalshi",
            max_capital="20", max_loss="20", model_hash="model-v1", config_hash="config-v1",
            readiness_hash="ready-v1",
        ), owner_id="owner-001", readiness_hash="ready-v1" if environment == "live" else None,
    )
    context = ExecutionContext(
        scope=active_scope, policy_version="policy-v1", strategy_version="strategy-v1", market_version="market-v1",
        runtime_live_enabled=True, autonomous=autonomous, mandate=mandate if autonomous else None,
        readiness_hash="ready-v1" if autonomous else None,
    )
    return store, active_scope, trade_intent, store.command_for_intent(trade_intent), claim, context


class TwinExecutionTests(unittest.TestCase):
    def test_acknowledgement_keeps_one_prepared_identity_and_never_becomes_fill(self):
        store, _, trade_intent, command, claim, context = reserved()
        writes = []

        def submit(prepared):
            writes.append((prepared.client_order_id, prepared.request_fingerprint))
            return {"acknowledgement": {"acknowledged": True, "venue_order_id": "venue-001"}}

        result = submit_claimed_command(
            store, command=command, intent=trade_intent, claim=claim, context=context, now=NOW, submit=submit,
        )
        self.assertEqual(result.disposition, SubmissionDisposition.ACKNOWLEDGED)
        self.assertEqual(result.command.state, CommandState.ACKNOWLEDGED)
        self.assertEqual(store.projection(context.scope.id).reserved_cash, Decimal("2.70"))
        self.assertEqual(len(writes), 1)
        self.assertTrue(writes[0][0].startswith("foresea-"))
        self.assertEqual(len(writes[0][1]), 64)

        duplicate = submit_claimed_command(
            store, command=command, intent=trade_intent, claim=claim, context=context, now=NOW, submit=submit,
        )
        self.assertEqual(duplicate.disposition, SubmissionDisposition.ALREADY_PROCESSED)
        self.assertEqual(len(writes), 1)

    def test_mutated_intent_and_revoked_mandate_make_zero_venue_writes(self):
        store, _, trade_intent, command, claim, context = reserved()
        writes = []
        mutated = replace(trade_intent, quantity=Decimal("6"))
        with self.assertRaises(ExecutionBlocked):
            submit_claimed_command(
                store, command=command, intent=mutated, claim=claim, context=context, now=NOW,
                submit=lambda _: writes.append("mutated"),
            )
        revoked = replace(context, mandate=revoke(context.mandate, owner_id="owner-001"))
        with self.assertRaises(ExecutionBlocked):
            submit_claimed_command(
                store, command=command, intent=trade_intent, claim=claim, context=revoked, now=NOW,
                submit=lambda _: writes.append("revoked"),
            )
        self.assertEqual(writes, [])

    def test_response_loss_transitions_to_unknown_and_never_retries(self):
        store, _, trade_intent, command, claim, context = reserved()
        writes = []

        def lost_response(_):
            writes.append("sent")
            raise TimeoutError("venue accepted before the connection dropped")

        with self.assertRaises(SubmissionUnknown):
            submit_claimed_command(
                store, command=command, intent=trade_intent, claim=claim, context=context, now=NOW,
                submit=lost_response,
            )
        current = store.command_for_intent(trade_intent)
        self.assertEqual(current.state, CommandState.SUBMISSION_UNKNOWN)
        repeat = submit_claimed_command(
            store, command=current, intent=trade_intent, claim=claim, context=context, now=NOW, submit=lost_response,
        )
        self.assertEqual(repeat.disposition, SubmissionDisposition.ALREADY_PROCESSED)
        self.assertEqual(writes, ["sent"])

    def test_confirmed_rejection_is_distinct_from_unknown_response(self):
        store, _, trade_intent, command, claim, context = reserved()
        rejected = submit_claimed_command(
            store, command=command, intent=trade_intent, claim=claim, context=context, now=NOW,
            submit=lambda _: {"acknowledgement": {"status": "rejected", "confirmed_rejection": True}},
        )
        self.assertEqual(rejected.disposition, SubmissionDisposition.REJECTED)
        self.assertEqual(rejected.command.state, CommandState.REJECTED)

    def test_live_runtime_gate_blocks_before_write_and_manual_confirmation_needs_no_strategy_gate(self):
        store, _, trade_intent, command, claim, context = reserved(environment="live")
        writes = []
        disabled = replace(context, runtime_live_enabled=False)
        with self.assertRaises(ExecutionBlocked):
            submit_claimed_command(
                store, command=command, intent=trade_intent, claim=claim, context=disabled, now=NOW,
                submit=lambda _: writes.append("live"),
            )
        self.assertEqual(writes, [])

        manual_store, _, manual_intent, manual_command, manual_claim, manual_context = reserved(autonomous=False)
        result = submit_claimed_command(
            manual_store, command=manual_command, intent=manual_intent, claim=manual_claim,
            context=manual_context, now=NOW,
            submit=lambda _: {"acknowledgement": {"acknowledged": True}},
        )
        self.assertEqual(result.command.state, CommandState.ACKNOWLEDGED)

    def test_legacy_shadow_wrapper_keeps_existing_callers_safe(self):
        active = approve(Mandate("mandate-legacy", "owner", "scope", "strategy", NOW + timedelta(days=1)), owner_id="owner")
        self.assertEqual(submit_authorized_command(active, now=NOW, live_enabled=False, submit=lambda: None), None)
        with self.assertRaises(ExecutionBlocked):
            submit_authorized_command(revoke(active, owner_id="owner"), now=NOW, live_enabled=False, submit=lambda: None)


if __name__ == "__main__":
    unittest.main()
