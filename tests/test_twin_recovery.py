import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import AccountScope, CommandState, InMemoryTwinStore, TradeIntent
from analyzing_llm_rationale.twin.models import ProposalAction
from analyzing_llm_rationale.twin.recovery import (
    RecoveryAction,
    VenueOrderLookup,
    recover_submission,
    recovery_action,
)

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def prepared_unknown():
    store = InMemoryTwinStore()
    scope = AccountScope(
        id="recovery-scope", owner_id="owner", venue="kalshi", venue_account_ref="account",
        environment="shadow", collateral_asset="USD", connection_ref="connection", account_epoch=1, created_at=NOW,
    )
    intent = TradeIntent(
        id="recovery-intent", account_scope_id=scope.id, account_epoch=1, instrument_id="kalshi:shadow:KXTEST",
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


if __name__ == "__main__":
    unittest.main()
