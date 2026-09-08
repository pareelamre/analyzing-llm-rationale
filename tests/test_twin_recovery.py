import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

from analyzing_llm_rationale.twin import (
    AccountScope,
    AccountSnapshot,
    CommandState,
    Completeness,
    InMemoryTwinStore,
    ReservationState,
    TradeIntent,
    TwinStoreError,
)
from analyzing_llm_rationale.twin.models import ProposalAction
from analyzing_llm_rationale.twin.recovery import (
    FillObservation,
    InMemoryLifecycleStore,
    LifecycleProjection,
    RecoveryAction,
    RecoveryBlocked,
    SettlementObservation,
    VenueOrderLookup,
    apply_lifecycle_observations,
    cancel_after_reconciliation,
    lookup_from_complete_account,
    reconcile_lifecycle,
    recover_submission,
    recovery_action,
    startup_recovery_action,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class DurableMemoryTwinStore(InMemoryTwinStore):
    durable = True


def prepared_unknown(venue="kalshi", *, environment="shadow", store=None):
    store = store or InMemoryTwinStore()
    scope = AccountScope(
        id="recovery-scope", owner_id="owner", venue=venue, venue_account_ref="account",
        environment=environment, collateral_asset="USD", connection_ref="connection", account_epoch=1, created_at=NOW,
    )
    intent = TradeIntent(
        id="recovery-intent", account_scope_id=scope.id, account_epoch=1,
        instrument_id=f"{venue}:{environment}:{'KXTEST' if venue == 'kalshi' else 'token-1'}",
        action=ProposalAction.BUY_YES, quantity=Decimal("2"), limit_price=Decimal("0.5"), time_in_force="IOC",
        forecast_id="forecast", exit_reason=None, policy_version="policy", strategy_version="strategy",
        market_version="market", fee_allowance=Decimal("0"), slippage_allowance=Decimal("0"),
        expires_at=NOW + timedelta(minutes=5), created_at=NOW,
    )
    store.register_account(scope, venue_available_cash=Decimal("10"), loss_limit=Decimal("10"))
    store.reserve_intent(intent, cash=Decimal("1"), max_loss=Decimal("1"), now=NOW)
    command = store.command_for_intent(intent)
    first = store.claim_command(command.id, worker_id="crashed-worker", now=NOW, lease_seconds=1)
    store.transition_command(command.id, target=CommandState.SUBMISSION_UNKNOWN, fence=first.fence, worker_id=first.worker_id)
    second = store.claim_command(command.id, worker_id="recovery-worker", now=NOW + timedelta(seconds=2), lease_seconds=30)
    return store, intent, store.command_for_intent(intent), second


def lookup(command, intent, *, found, complete=True, fingerprint=None, offset=3):
    return VenueOrderLookup(
        account_scope_id=command.scope_id, instrument_id=intent.instrument_id, client_order_id=command.client_order_id,
        request_fingerprint=command.request_fingerprint if fingerprint is None else fingerprint,
        complete=complete, order_found=found if complete else None, observed_at=NOW + timedelta(seconds=offset),
    )


def fill(command, intent, fill_id, quantity, *, version=1, offset=3):
    return FillObservation(
        fill_id, version, "venue-order-1", command.client_order_id,
        intent.instrument_id, Decimal(quantity), NOW + timedelta(seconds=offset - 1),
        NOW + timedelta(seconds=offset),
    )


def acknowledged():
    store, intent, command, claim = prepared_unknown()
    recover_submission(
        store, command=command, intent=intent, claim=claim,
        now=NOW + timedelta(seconds=3), lookups=[lookup(command, intent, found=True)],
    )
    return store, intent, store.command_for_intent(intent), claim


def account_snapshot(command, *, orders=(), fills=(), completeness=Completeness.COMPLETE, divergence=False):
    return AccountSnapshot(
        command.scope_id, 1, NOW, completeness,
        Decimal("9"), Decimal("10"), Decimal("1"), Decimal("10"), (),
        Decimal("0"), Decimal("0"), Decimal("10"), (), tuple(orders), tuple(fills), (),
        (), divergence, ("external activity",) if divergence else (),
    )


def settlement(command, intent, *, version, amount, final, offset=0, settlement_id="settle-1"):
    return SettlementObservation(
        settlement_id, version, "venue-order-1", command.client_order_id,
        intent.instrument_id, Decimal(amount), final, NOW,
        NOW + timedelta(seconds=offset),
    )


class TwinRecoveryTests(unittest.TestCase):
    def test_unknown_submission_never_retries_without_reconciliation(self):
        self.assertEqual(recovery_action("submission_unknown", None), "pause_and_reconcile")
        self.assertEqual(recovery_action("submission_unknown", True), "reconcile")
        self.assertEqual(recovery_action("filled", None), "terminal")

    def test_found_order_is_acknowledged_using_the_prepared_identity(self):
        store, intent, command, claim = prepared_unknown()
        result = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=True)],
        )
        self.assertEqual(result.action, RecoveryAction.RECONCILED)
        self.assertEqual(result.command.state, CommandState.ACKNOWLEDGED)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("1"))

    def test_crash_before_send_releases_only_after_confirmed_absence(self):
        store = InMemoryTwinStore()
        scope = AccountScope(
            "before-send", "owner", "kalshi", "account", "shadow", "USD",
            "connection", 1, NOW,
        )
        intent = TradeIntent(
            "before-send-intent", scope.id, 1, "kalshi:shadow:KXTEST",
            ProposalAction.BUY_YES, Decimal("1"), Decimal("0.5"), "IOC",
            "forecast", None, "p", "s", "m", Decimal("0"), Decimal("0"),
            NOW + timedelta(minutes=5), NOW,
        )
        store.register_account(scope, venue_available_cash=Decimal("2"), loss_limit=Decimal("2"))
        store.reserve_intent(intent, cash=Decimal("0.5"), max_loss=Decimal("0.5"), now=NOW)
        command = store.command_for_intent(intent)
        claim = store.claim_command(command.id, worker_id="replacement", now=NOW, lease_seconds=30)
        command = store.command_for_intent(intent)
        result = recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=3),
            lookups=[
                lookup(command, intent, found=False, offset=2),
                lookup(command, intent, found=False, offset=3),
            ],
        )
        self.assertEqual(result.action, RecoveryAction.CONFIRMED_ABSENT)

    def test_crash_after_acceptance_before_receipt_uses_venue_identity(self):
        store, intent, command, claim = prepared_unknown()
        result = recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=True)],
        )
        self.assertEqual(result.command.client_order_id, command.client_order_id)
        self.assertEqual(result.command.state, CommandState.ACKNOWLEDGED)

    def test_account_reconnect_epoch_blocks_recovery(self):
        store, intent, command, claim = prepared_unknown()
        reconnected = replace(store.account_scope(command.scope_id), account_epoch=2)
        with patch.object(store, "account_scope", return_value=reconnected):
            with self.assertRaisesRegex(RecoveryBlocked, "account reconnect"):
                recover_submission(
                    store, command=command, intent=intent, claim=claim,
                    now=NOW + timedelta(seconds=3),
                    lookups=[lookup(command, intent, found=True)],
                )

    def test_incomplete_or_identity_mismatch_holds_reservation_for_operator(self):
        store, intent, command, claim = prepared_unknown()
        incomplete = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=False, complete=False)],
        )
        self.assertEqual(incomplete.action, RecoveryAction.OPERATOR_ATTENTION)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("1"))

        mismatch = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=True, fingerprint="wrong")],
        )
        self.assertEqual(mismatch.action, RecoveryAction.OPERATOR_ATTENTION)
        recovered = recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=4),
            lookups=[
                lookup(command, intent, found=False, complete=False, offset=3),
                lookup(command, intent, found=True, offset=4),
            ],
        )
        self.assertEqual(recovered.action, RecoveryAction.RECONCILED)


    def test_repeated_or_future_absence_observations_do_not_release_capital(self):
        store, intent, command, claim = prepared_unknown()
        repeated = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=False), lookup(command, intent, found=False)],
        )
        self.assertEqual(repeated.action, RecoveryAction.OPERATOR_ATTENTION)
        future = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=False, offset=4)],
        )
        self.assertEqual(future.action, RecoveryAction.OPERATOR_ATTENTION)

    def test_two_complete_absence_observations_release_once_without_new_identity(self):
        store, intent, command, claim = prepared_unknown()
        result = recover_submission(
            store, command=command, intent=intent, claim=claim, now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=False, offset=2), lookup(command, intent, found=False, offset=3)],
        )
        self.assertEqual(result.action, RecoveryAction.CONFIRMED_ABSENT)
        self.assertTrue(result.reservation_released)
        self.assertEqual(result.command.state, CommandState.REJECTED)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("0"))

    def test_polymarket_absence_requires_three_observations_five_seconds_apart(self):
        store, intent, command, claim = prepared_unknown("polymarket")
        too_few = recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=14),
            lookups=[
                lookup(command, intent, found=False, offset=3),
                lookup(command, intent, found=False, offset=8),
            ],
        )
        self.assertEqual(too_few.action, RecoveryAction.OPERATOR_ATTENTION)
        result = recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=14),
            lookups=[
                lookup(command, intent, found=False, offset=3),
                lookup(command, intent, found=False, offset=8),
                lookup(command, intent, found=False, offset=13),
            ],
        )
        self.assertEqual(result.action, RecoveryAction.CONFIRMED_ABSENT)

    def test_fill_versions_are_idempotent_and_out_of_order_updates_do_not_double_count(self):
        base = LifecycleProjection(
            "command-1", "scope-1", "venue-order-1", "client-1",
            "kalshi:shadow:KXTEST", Decimal("3"),
        )
        second = FillObservation(
            "fill-1", 2, "venue-order-1", "client-1", base.instrument_id,
            Decimal("2"), NOW, NOW + timedelta(seconds=2),
        )
        first = FillObservation(
            "fill-1", 1, "venue-order-1", "client-1", base.instrument_id,
            Decimal("1"), NOW, NOW + timedelta(seconds=1),
        )
        updated = apply_lifecycle_observations(
            base, fills=[second], observed_at=NOW + timedelta(seconds=2),
        )
        replayed = apply_lifecycle_observations(
            updated, fills=[second, first], observed_at=NOW + timedelta(seconds=3),
        )
        self.assertEqual(replayed.filled_quantity, Decimal("2"))
        self.assertEqual(replayed.revision, updated.revision)
        with self.assertRaisesRegex(RecoveryBlocked, "conflicting"):
            apply_lifecycle_observations(
                updated,
                fills=[FillObservation(
                    "fill-1", 2, "venue-order-1", "client-1", base.instrument_id,
                    Decimal("1.5"), NOW, NOW + timedelta(seconds=2),
                )],
                observed_at=NOW + timedelta(seconds=3),
            )

    def test_lifecycle_storage_round_trip_rejects_schema_and_boolean_coercion(self):
        base = LifecycleProjection(
            "command-1", "scope-1", "venue-order-1", "client-1",
            "kalshi:shadow:KXTEST", Decimal("1"),
        )
        payload = base.to_storage()
        self.assertEqual(LifecycleProjection.from_storage(payload), base)
        payload["schema_version"] = 2
        with self.assertRaisesRegex(RecoveryBlocked, "schema"):
            LifecycleProjection.from_storage(payload)
        with self.assertRaisesRegex(RecoveryBlocked, "revision"):
            LifecycleProjection(
                "command-1", "scope-1", "venue-order-1", "client-1",
                "kalshi:shadow:KXTEST", Decimal("1"), revision=True,
            )
        malformed = SettlementObservation(
            "settle-1", 1, "venue-order-1", "client-1", base.instrument_id,
            Decimal("1"), False, NOW, NOW,
        ).to_storage()
        malformed["final"] = "false"
        with self.assertRaisesRegex(RecoveryBlocked, "malformed"):
            SettlementObservation.from_storage(malformed)

    def test_cancel_fill_race_preserves_late_fills_and_never_reuses_order_identity(self):
        store, intent, command, claim = acknowledged()
        writes = []
        response = cancel_after_reconciliation(
            store, command=command, intent=intent, claim=claim,
            lookup=lookup(command, intent, found=True),
            cancel=lambda current: writes.append(current.client_order_id) or {"status": "cancelled"},
            now=NOW + timedelta(seconds=3),
        )
        self.assertEqual(response["status"], "cancelled")
        self.assertEqual(store.command_for_intent(intent).state, CommandState.CANCEL_REQUESTED)
        ledger = InMemoryLifecycleStore()
        reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id="venue-order-1", fills=[fill(command, intent, "fill-1", "1")],
            order_status="cancelled", observed_at=NOW + timedelta(seconds=3),
        )
        self.assertEqual(store.command_for_intent(intent).state, CommandState.PARTIALLY_FILLED)
        projection = reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id="venue-order-1", fills=[fill(command, intent, "fill-2", "1", offset=4)],
            order_status="cancelled", observed_at=NOW + timedelta(seconds=4),
        )
        self.assertEqual(projection.filled_quantity, Decimal("2"))
        self.assertEqual(writes, [command.client_order_id])

    def test_crash_after_partial_fill_reacquires_fence_and_applies_late_fill(self):
        store, intent, command, first_claim = acknowledged()
        ledger = InMemoryLifecycleStore()
        reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=first_claim,
            venue_order_id="venue-order-1",
            fills=[fill(command, intent, "fill-1", "1")],
            order_status="partially_filled", observed_at=NOW + timedelta(seconds=3),
        )
        partial = store.command_for_intent(intent)
        self.assertEqual(partial.state, CommandState.PARTIALLY_FILLED)
        replacement = store.claim_command(
            partial.id, worker_id="replacement", now=NOW + timedelta(seconds=33),
            lease_seconds=30,
        )
        self.assertIsNotNone(replacement)
        completed = reconcile_lifecycle(
            store, ledger, command=partial, intent=intent, claim=replacement,
            venue_order_id="venue-order-1",
            fills=[fill(command, intent, "fill-2", "1", offset=34)],
            order_status="filled", observed_at=NOW + timedelta(seconds=34),
        )
        self.assertEqual(completed.filled_quantity, Decimal("2"))
        self.assertEqual(store.command_for_intent(intent).state, CommandState.FILLED)

    def test_cancel_failure_requires_a_fresh_reconciliation_before_retry(self):
        store, intent, command, claim = acknowledged()
        with self.assertRaisesRegex(RecoveryBlocked, "reconcile before another attempt"):
            cancel_after_reconciliation(
                store, command=command, intent=intent, claim=claim,
                lookup=lookup(command, intent, found=True),
                cancel=lambda _: (_ for _ in ()).throw(TimeoutError("lost")),
                now=NOW + timedelta(seconds=3),
            )
        self.assertEqual(store.command_for_intent(intent).state, CommandState.CANCEL_REQUESTED)
        with self.assertRaisesRegex(RecoveryBlocked, "matching live-order"):
            cancel_after_reconciliation(
                store, command=command, intent=intent, claim=claim,
                lookup=lookup(command, intent, found=False, offset=4), cancel=lambda _: {},
                now=NOW + timedelta(seconds=4),
            )

    def test_cancel_rejects_stale_lookup_and_unconfirmed_response(self):
        store, intent, command, claim = acknowledged()
        with self.assertRaisesRegex(RecoveryBlocked, "stale"):
            cancel_after_reconciliation(
                store, command=command, intent=intent, claim=claim,
                lookup=lookup(command, intent, found=True, offset=3), cancel=lambda _: {},
                now=NOW + timedelta(seconds=9),
            )
        with self.assertRaisesRegex(RecoveryBlocked, "unconfirmed"):
            cancel_after_reconciliation(
                store, command=command, intent=intent, claim=claim,
                lookup=lookup(command, intent, found=True, offset=4), cancel=lambda _: {},
                now=NOW + timedelta(seconds=4),
            )

    def test_provisional_final_and_corrective_settlement_revisions_are_retained(self):
        base = LifecycleProjection(
            "command-1", "scope-1", "venue-order-1", "client-1",
            "kalshi:shadow:KXTEST", Decimal("1"),
        )
        provisional = SettlementObservation(
            "settle-1", 1, "venue-order-1", "client-1", base.instrument_id,
            Decimal("1"), False, NOW, NOW,
        )
        current = apply_lifecycle_observations(base, settlements=[provisional], observed_at=NOW)
        self.assertIsNone(current.settled_amount)
        final = SettlementObservation(
            "settle-1", 2, "venue-order-1", "client-1", base.instrument_id,
            Decimal("1"), True, NOW, NOW + timedelta(seconds=1),
        )
        current = apply_lifecycle_observations(
            current, settlements=[final], observed_at=NOW + timedelta(seconds=1),
        )
        self.assertEqual(current.settled_amount, Decimal("1"))
        correction = SettlementObservation(
            "settle-1", 3, "venue-order-1", "client-1", base.instrument_id,
            Decimal("0.9"), True, NOW, NOW + timedelta(seconds=2),
        )
        current = apply_lifecycle_observations(
            current, settlements=[correction], observed_at=NOW + timedelta(seconds=2),
        )
        self.assertEqual(current.settled_amount, Decimal("0.9"))

        with self.assertRaisesRegex(RecoveryBlocked, "settlement does not match"):
            apply_lifecycle_observations(
                current,
                settlements=[SettlementObservation(
                    "other", 1, "another-order", "client-1", base.instrument_id,
                    Decimal("1"), True, NOW, NOW + timedelta(seconds=3),
                )],
                observed_at=NOW + timedelta(seconds=3),
            )

    def test_only_final_settlement_releases_reservation_and_corrections_are_idempotent(self):
        store, intent, command, claim = acknowledged()
        ledger = InMemoryLifecycleStore()
        provisional = settlement(command, intent, version=1, amount="1", final=False)
        reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id="venue-order-1", settlements=[provisional],
            order_status="open", observed_at=NOW,
        )
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("1"))
        final = settlement(command, intent, version=2, amount="1", final=True, offset=1)
        reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id="venue-order-1", settlements=[final],
            order_status="cancelled", observed_at=NOW + timedelta(seconds=1),
        )
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("0"))
        self.assertEqual(
            store.reservation(command.scope_id, command.reservation_id).state,
            ReservationState.SETTLED,
        )
        final_ref = store.reservation(
            command.scope_id, command.reservation_id,
        ).reconciliation_ref
        correction = settlement(command, intent, version=3, amount="0.9", final=True, offset=2)
        reconciled = reconcile_lifecycle(
            store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id="venue-order-1", settlements=[correction],
            order_status="cancelled", observed_at=NOW + timedelta(seconds=2),
        )
        self.assertEqual(reconciled.settled_amount, Decimal("0.9"))
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal("0"))
        self.assertNotEqual(
            store.reservation(command.scope_id, command.reservation_id).reconciliation_ref,
            final_ref,
        )

    def test_live_lifecycle_refuses_process_local_evidence_storage(self):
        store, intent, command, claim = prepared_unknown(
            environment="live", store=DurableMemoryTwinStore(),
        )
        recover_submission(
            store, command=command, intent=intent, claim=claim,
            now=NOW + timedelta(seconds=3),
            lookups=[lookup(command, intent, found=True)],
        )
        with self.assertRaisesRegex(RecoveryBlocked, "durable storage"):
            reconcile_lifecycle(
                store, InMemoryLifecycleStore(), command=command, intent=intent,
                claim=claim, venue_order_id="venue-order-1", order_status="open",
                observed_at=NOW + timedelta(seconds=3),
            )

    def test_unsubmitted_reservation_cannot_be_settled(self):
        store = InMemoryTwinStore()
        scope = AccountScope(
            "unsubmitted", "owner", "kalshi", "account", "shadow", "USD",
            "connection", 1, NOW,
        )
        intent = TradeIntent(
            "unsubmitted-intent", scope.id, 1, "kalshi:shadow:KXTEST",
            ProposalAction.BUY_YES, Decimal("1"), Decimal("0.5"), "IOC",
            "forecast", None, "p", "s", "m", Decimal("0"), Decimal("0"),
            NOW + timedelta(minutes=5), NOW,
        )
        store.register_account(scope, venue_available_cash=Decimal("2"), loss_limit=Decimal("2"))
        reservation = store.reserve_intent(
            intent, cash=Decimal("0.5"), max_loss=Decimal("0.5"), now=NOW,
        )
        with self.assertRaisesRegex(TwinStoreError, "not eligible"):
            store.settle_reservation(scope.id, reservation.id, settlement_ref="settlement")

    def test_stale_recovery_worker_cannot_progress_after_losing_fence(self):
        store, intent, command, old_claim = prepared_unknown()
        new_claim = store.claim_command(
            command.id, worker_id="new-worker", now=NOW + timedelta(seconds=33),
            lease_seconds=30,
        )
        self.assertIsNotNone(new_claim)
        with self.assertRaisesRegex(RecoveryBlocked, "no longer owns"):
            reconcile_lifecycle(
                store, InMemoryLifecycleStore(), command=command, intent=intent,
                claim=old_claim, venue_order_id="venue-order-1",
                order_status="open", observed_at=NOW + timedelta(seconds=33),
            )

    def test_startup_never_treats_an_expired_lease_as_no_order(self):
        store, intent, command, _ = prepared_unknown()
        self.assertEqual(
            startup_recovery_action(command, lease_expired=True),
            RecoveryAction.OPERATOR_ATTENTION,
        )
        reserved_store = InMemoryTwinStore()
        active_scope = AccountScope(
            "startup", "owner", "kalshi", "account", "shadow", "USD",
            "connection", 1, NOW,
        )
        reserved_store.register_account(
            active_scope, venue_available_cash=Decimal("1"), loss_limit=Decimal("1"),
        )
        reserved_intent = TradeIntent(
            "startup-intent", active_scope.id, 1, "kalshi:shadow:KX", ProposalAction.BUY_YES,
            Decimal("1"), Decimal("0.5"), "IOC", "forecast", None, "p", "s", "m",
            Decimal("0"), Decimal("0"), NOW + timedelta(minutes=1), NOW,
        )
        reserved_store.reserve_intent(
            reserved_intent, cash=Decimal("0.5"), max_loss=Decimal("0.5"), now=NOW,
        )
        self.assertEqual(
            startup_recovery_action(
                reserved_store.command_for_intent(reserved_intent), lease_expired=False,
            ),
            RecoveryAction.RECONCILED,
        )
        for state in (CommandState.FILLED, CommandState.CANCELLED):
            self.assertEqual(
                startup_recovery_action(replace(command, state=state), lease_expired=True),
                RecoveryAction.RECONCILED,
            )

    def test_complete_account_generation_proves_presence_or_absence_by_exact_identity(self):
        _, intent, command, _ = prepared_unknown()
        found = lookup_from_complete_account(
            account_snapshot(command, orders=[{
                "order_id": "venue-order-1", "client_order_id": command.client_order_id,
                "ticker": "KXTEST",
            }]),
            command=command, intent=intent, observed_at=NOW,
        )
        absent = lookup_from_complete_account(
            account_snapshot(command), command=command, intent=intent, observed_at=NOW,
        )
        self.assertTrue(found.order_found)
        self.assertFalse(absent.order_found)
        incomplete = lookup_from_complete_account(
            account_snapshot(command, completeness=Completeness.INCOMPLETE),
            command=command, intent=intent, observed_at=NOW,
        )
        drifted = lookup_from_complete_account(
            account_snapshot(command, divergence=True),
            command=command, intent=intent, observed_at=NOW,
        )
        self.assertFalse(incomplete.complete)
        self.assertFalse(drifted.complete)
        with self.assertRaisesRegex(RecoveryBlocked, "observation time"):
            lookup_from_complete_account(
                account_snapshot(command), command=command, intent=intent,
                observed_at=NOW + timedelta(seconds=1),
            )
        with self.assertRaisesRegex(RecoveryBlocked, "another instrument"):
            lookup_from_complete_account(
                account_snapshot(command, fills=[{
                    "fill_id": "fill-1", "client_order_id": command.client_order_id,
                    "ticker": "KXOTHER",
                }]),
                command=command, intent=intent, observed_at=NOW,
            )


if __name__ == "__main__":
    unittest.main()
