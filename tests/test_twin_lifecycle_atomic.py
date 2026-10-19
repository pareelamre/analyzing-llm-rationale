"""Account-fenced lifecycle evidence and cash must commit as one unit."""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest import mock

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin import recovery
from analyzing_llm_rationale.twin.models import CommandState
from analyzing_llm_rationale.twin.recovery import (
    DatastoreLifecycleStore,
    InMemoryLifecycleStore,
    RecoveryBlocked,
    reconcile_lifecycle,
    recover_submission,
)
from analyzing_llm_rationale.twin.store import (
    DatastoreTwinStore,
    InMemoryTwinStore,
    ReservationState,
    TwinStoreError,
)
from tests.test_twin_recovery import NOW, fill, lookup, prepared_unknown, settlement


class AtomicLifecycleTests(unittest.TestCase):
    backends = ('memory', 'sql')

    def pair(self, backend, *, acknowledge=True):
        if backend == 'memory':
            store, ledger = InMemoryTwinStore(), InMemoryLifecycleStore()
        else:
            client = Client(':memory:')
            self.addCleanup(client.close)
            store, ledger = DatastoreTwinStore(client), DatastoreLifecycleStore(client)
        store, intent, command, claim = prepared_unknown(store=store)
        if acknowledge:
            recover_submission(store, command=command, intent=intent, claim=claim,
                now=NOW+timedelta(seconds=3), lookups=[lookup(command, intent, found=True)], clock=lambda: NOW+timedelta(seconds=3))
        return store, ledger, intent, store.command_for_intent(intent), claim

    def run_reconcile(self, values, *, at=None, **kwargs):
        store, ledger, intent, command, claim = values
        with mock.patch.object(recovery, 'datetime', wraps=datetime) as dates:
            dates.now.side_effect = (lambda *_: at[0]) if isinstance(at, list) else None
            dates.now.return_value = at or NOW+timedelta(seconds=3)
            return reconcile_lifecycle(store, ledger, command=command, intent=intent,
                claim=claim, venue_order_id='venue-order-1',
                observed_at=NOW+timedelta(seconds=3), **kwargs)

    def assert_held_without_evidence(self, values):
        store, ledger, intent, command, _ = values
        self.assertIsNone(ledger.load(command.scope_id, command.id))
        self.assertEqual(store.command_for_intent(intent).state, CommandState.ACKNOWLEDGED)
        self.assertEqual(store.projection(command.scope_id).reserved_cash, Decimal('1'))

    def test_expired_processing_lease_cannot_backdate_its_observation(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                with self.assertRaises(RecoveryBlocked):
                    self.run_reconcile(values, at=NOW+timedelta(seconds=33), order_status='open')
                self.assert_held_without_evidence(values)

    def test_transition_failure_rolls_back_evidence(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                with mock.patch.object(values[0], 'transition_command', side_effect=TwinStoreError('injected')):
                    with self.assertRaises(RecoveryBlocked):
                        self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '1')])
                self.assert_held_without_evidence(values)

    def test_cancellation_rolls_back_and_leaves_store_usable(self):
        class Cancelled(BaseException):
            pass
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                with mock.patch.object(values[0], 'transition_command', side_effect=Cancelled()):
                    with self.assertRaises(Cancelled):
                        self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '1')])
                self.assert_held_without_evidence(values)
                result = self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '1')])
                self.assertEqual(result.filled_quantity, 1)

    def test_failure_after_settlement_rolls_back_cash_command_and_evidence(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                original = values[0].settle_reservation
                def fail_after_write(*args, _original=original, **kwargs):
                    _original(*args, **kwargs)
                    raise TwinStoreError('injected after settlement')
                with mock.patch.object(values[0], 'settle_reservation', side_effect=fail_after_write):
                    with self.assertRaises(RecoveryBlocked):
                        self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '2')],
                            settlements=[settlement(values[3], values[2], version=1, amount='2', final=True)],
                            terminal_evidence_complete=True, order_status='filled')
                self.assert_held_without_evidence(values)

    def test_lease_expiring_during_save_rolls_back_evidence(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                now = [NOW+timedelta(seconds=3)]
                original = values[1].save
                def advance_after_save(*args, _original=original, _now=now, **kwargs):
                    result = _original(*args, **kwargs)
                    _now[0] = NOW+timedelta(seconds=33)
                    return result
                with mock.patch.object(values[1], 'save', side_effect=advance_after_save):
                    with self.assertRaises(RecoveryBlocked):
                        self.run_reconcile(values, at=now, order_status='open')
                self.assert_held_without_evidence(values)

    def test_filled_status_without_unique_fills_does_not_invent_inventory(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                result = self.run_reconcile(values, order_status='filled')
                self.assertEqual(result.filled_quantity, 0)
                self.assertEqual(values[0].command_for_intent(values[2]).state, CommandState.ACKNOWLEDGED)
                self.assertEqual(values[0].projection(values[3].scope_id).reserved_cash, Decimal('1'))

    def test_future_observation_does_not_commit(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                with self.assertRaises(RecoveryBlocked):
                    self.run_reconcile(values, at=NOW+timedelta(seconds=2), order_status='open')
                self.assert_held_without_evidence(values)

    def test_complete_zero_fill_rejection_releases_once(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend, acknowledge=False)
                for _ in range(2):
                    self.run_reconcile(values, order_status='rejected', terminal_evidence_complete=True)
                self.assertEqual(values[0].projection(values[3].scope_id).reserved_cash, 0)
                self.assertEqual(values[0].reservation(values[3].scope_id,
                    values[3].reservation_id).state, ReservationState.RELEASED)

    def test_incomplete_rejection_does_not_release_unknown_exposure(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend, acknowledge=False)
                with self.assertRaisesRegex(RecoveryBlocked, 'complete zero-fill'):
                    self.run_reconcile(values, order_status='rejected')
                self.assertIsNone(values[1].load(values[3].scope_id, values[3].id))
                self.assertEqual(values[0].projection(values[3].scope_id).reserved_cash, 1)
                self.assertEqual(values[0].command_for_intent(values[2]).state, CommandState.SUBMISSION_UNKNOWN)

    def test_successful_full_fill_and_settlement_commit_once(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                store, ledger, intent, command, _ = values
                for _ in range(2):
                    result = self.run_reconcile(values, order_status='filled',
                        fills=[fill(command,intent,'f','2')], terminal_evidence_complete=True,
                        settlements=[settlement(command,intent,version=1,amount='2',final=True)])
                self.assertEqual(result.filled_quantity, 2)
                self.assertEqual(ledger.load(command.scope_id,command.id), result)
                self.assertEqual(store.command_for_intent(intent).state, CommandState.FILLED)
                self.assertEqual(store.reservation(command.scope_id,command.reservation_id).state, ReservationState.SETTLED)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 0)


class IndependentLifecycleStoreTests(unittest.TestCase):
    def test_independent_sql_clients_are_not_a_coherent_transaction(self):
        case = AtomicLifecycleTests(methodName='runTest')
        self.addCleanup(case.doCleanups)
        values = case.pair('sql')
        client = Client(':memory:')
        self.addCleanup(client.close)
        other = (values[0], DatastoreLifecycleStore(client), *values[2:])
        with self.assertRaisesRegex(RecoveryBlocked, 'coherent account transaction'):
            case.run_reconcile(other, order_status='open')
        case.assert_held_without_evidence(values)
