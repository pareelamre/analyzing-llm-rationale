"""Regression cases from the native execution review; no venue transport."""
import unittest
from contextlib import ExitStack, closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.models import CommandState
from analyzing_llm_rationale.twin.recovery import (
    InMemoryLifecycleStore,
    RecoveryAction,
    reconcile_lifecycle,
    recover_submission,
)
from analyzing_llm_rationale.twin.store import DatastoreTwinStore, TwinStoreError
from tests.test_twin_recovery import NOW, acknowledged, lookup, prepared_unknown, settlement


class DispatchSafetyTests(unittest.TestCase):
    def test_durable_colon_scope_and_recent_pages_use_current_records(self):
        with TemporaryDirectory() as root, mock.patch.dict('os.environ', {'FORESEA_DATASTORE_BACKEND': 'sql'}), closing(
            Client(path=str(Path(root)/'state.sqlite3')),
        ) as db:
            store, trade, old, _ = prepared_unknown(store=DatastoreTwinStore(db))
            scope = store.account_scope(old.scope_id)
            newer_scope = replace(scope, id='venue:live:account', created_at=NOW+timedelta(seconds=1))
            store.register_account(newer_scope, venue_available_cash=Decimal('10'), loss_limit=Decimal('10'))
            self.assertEqual(store.account_scopes(scope.owner_id, limit=1)[0], newer_scope)
            first = replace(trade, account_scope_id=newer_scope.id)
            store.reserve_intent(first, cash=Decimal('1'), max_loss=Decimal('1'), now=NOW)
            second = replace(first, forecast_id='newer-forecast', created_at=NOW+timedelta(seconds=1))
            store.reserve_intent(second, cash=Decimal('1'), max_loss=Decimal('1'), now=NOW+timedelta(seconds=1))
            command = store.command_for_intent(second)
            self.assertEqual(store.commands(newer_scope.id, limit=1)[0].id, command.id)
            claim = store.claim_command(command.id, worker_id='fixture', now=NOW+timedelta(seconds=1))
            self.assertTrue(store.begin_dispatch(command.id, worker_id=claim.worker_id,
                fence=claim.fence, now=NOW+timedelta(seconds=1)))
            store.transition_command(command.id, target=CommandState.REJECTED,
                fence=claim.fence, worker_id=claim.worker_id)
            self.assertEqual(store.projection(newer_scope.id).reserved_cash, Decimal('1'))

    def test_release_after_settlement_cannot_subtract_twice_in_either_store(self):
        for durable in (False, True):
            with self.subTest(durable=durable), ExitStack() as stack:
                target = None
                if durable:
                    stack.enter_context(mock.patch.dict('os.environ', {'FORESEA_DATASTORE_BACKEND': 'sql'}))
                    root = stack.enter_context(TemporaryDirectory())
                    db = Client(path=str(Path(root)/'state.sqlite3'))
                    stack.callback(db.close)
                    target = DatastoreTwinStore(db)
                store, intent, command, claim = prepared_unknown(store=target)
                store.settle_reservation(command.scope_id, command.reservation_id, settlement_ref='fixture-final')
                before = store.projection(command.scope_id)
                with self.assertRaises(TwinStoreError):
                    store.release_reservation(command.scope_id, command.reservation_id, confirmed_no_order=True)
                self.assertEqual(store.projection(command.scope_id), before)
                self.assertEqual(before.reserved_cash, 0)

    def test_absence_cannot_release_a_command_whose_dispatch_started(self):
        store, intent, command, claim = prepared_unknown()
        # Simulate the persisted marker from a crashed transport, not a missing order.
        store._commands[command.id] = replace(command, dispatch_started=True, dispatch_fence=1)
        result = recover_submission(store, command=command, intent=intent, claim=claim,
            now=NOW+timedelta(seconds=5), lookups=[
                lookup(command, intent, found=False, offset=3),
                lookup(command, intent, found=False, offset=5)])
        self.assertEqual(result.action, RecoveryAction.OPERATOR_ATTENTION)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal('1'))

    def test_final_settlement_on_open_order_does_not_free_exposure(self):
        store, intent, command, claim = acknowledged()
        result = reconcile_lifecycle(store, InMemoryLifecycleStore(), command=command,
            intent=intent, claim=claim, venue_order_id='venue-order-1',
            settlements=[settlement(command, intent, version=1, amount='1', final=True)],
            order_status='open', observed_at=NOW)
        self.assertFalse(result.settlement_final)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal('1'))

    def test_cancel_requires_complete_terminal_evidence_to_free_zero_fill_hold(self):
        store, intent, command, claim = acknowledged()
        ledger = InMemoryLifecycleStore()
        reconcile_lifecycle(store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id='venue-order-1', order_status='cancelled', observed_at=NOW)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal('1'))
        reconcile_lifecycle(store, ledger, command=command, intent=intent, claim=claim,
            venue_order_id='venue-order-1', order_status='cancelled',
            terminal_evidence_complete=True, observed_at=NOW+timedelta(seconds=1))
        self.assertEqual(store.projection(command.scope_id).reserved_cash, 0)
        self.assertEqual(store.command_for_intent(intent).state, CommandState.CANCELLED)
