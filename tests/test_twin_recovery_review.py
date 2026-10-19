"""Behavior regressions for the single T15 independent review."""
import sqlite3
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest import mock

from analyzing_llm_rationale.datastore_sql import Client, Entity
from analyzing_llm_rationale.twin import recovery
from analyzing_llm_rationale.twin.account_store import (
    DatastoreAccountSnapshotStore,
    InMemoryAccountSnapshotStore,
)
from analyzing_llm_rationale.twin.models import CommandState
from analyzing_llm_rationale.twin.recovery import (
    RecoveryBlocked,
    cancel_after_reconciliation,
    reconcile_lifecycle,
    recover_submission,
)
from analyzing_llm_rationale.twin.store import TwinStoreError
from tests import test_twin_lifecycle_atomic as atomic
from tests.test_twin_recovery import NOW, account_snapshot, fill, lookup, settlement


class LifecycleReviewTests(unittest.TestCase):
    backends = ('memory', 'sql')
    pair = atomic.AtomicLifecycleTests.pair
    run_reconcile = atomic.AtomicLifecycleTests.run_reconcile
    assert_held_without_evidence = atomic.AtomicLifecycleTests.assert_held_without_evidence

    def test_final_filled_status_with_missing_or_partial_fills_is_quarantined(self):
        for backend in self.backends:
            for quantity in ('0', '1'):
                with self.subTest(backend=backend, quantity=quantity):
                    values = self.pair(backend)
                    observed = [] if quantity == '0' else [fill(values[3], values[2], 'fill', quantity)]
                    with self.assertRaisesRegex(RecoveryBlocked, 'filled_status_missing_fills'):
                        self.run_reconcile(values, order_status='filled', fills=observed,
                            settlements=[settlement(values[3], values[2], version=1, amount='1', final=True)],
                            terminal_evidence_complete=True)
                    self.assert_held_without_evidence(values)
                    attention = values[0].recovery_attention(values[3].scope_id)
                    self.assertEqual(attention[0]['reason'], 'filled_status_missing_fills')
                    self.assertEqual(len(attention[0]['evidence']['fills']), len(observed))

    def test_quarantine_pauses_new_reservations_and_already_reserved_dispatch(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                store, _, intent, command, _ = values
                other = replace(intent, id='pending', forecast_id='other-forecast')
                store.reserve_intent(other, cash=Decimal('1'), max_loss=Decimal('1'), now=NOW)
                pending = store.command_for_intent(other)
                claim = store.claim_command(pending.id, worker_id='pending', now=NOW, lease_seconds=30)
                with self.assertRaises(RecoveryBlocked):
                    self.run_reconcile(values, order_status='filled', terminal_evidence_complete=True)
                with self.assertRaisesRegex(TwinStoreError, 'paused'):
                    store.reserve_intent(replace(intent, forecast_id='third'), cash=Decimal('1'),
                        max_loss=Decimal('1'), now=NOW)
                with self.assertRaisesRegex(TwinStoreError, 'paused'):
                    store.begin_dispatch(pending.id, worker_id=claim.worker_id, fence=claim.fence,
                        now=NOW+timedelta(seconds=3))
                self.assertFalse(store.command_for_intent(other).dispatch_started)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 2)

    def test_mutated_command_cannot_release_another_reservation(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                store, ledger, intent, command, claim = values
                victim = replace(intent, id='victim', forecast_id='victim-forecast')
                store.reserve_intent(victim, cash=Decimal('1'), max_loss=Decimal('1'), now=NOW)
                other = store.command_for_intent(victim)
                store.claim_command(other.id, worker_id='victim', now=NOW)
                forged = replace(command, reservation_id=other.reservation_id)
                with self.assertRaises(RecoveryBlocked):
                    self.run_reconcile((store, ledger, intent, forged, claim), order_status='filled',
                        fills=[fill(command, intent, 'f', '2')], terminal_evidence_complete=True,
                        settlements=[settlement(command, intent, version=1, amount='2', final=True)])
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 2)
                self.assertIsNone(ledger.load(command.scope_id, command.id))
                self.assertEqual(store.recovery_attention(command.scope_id), ())

    def test_lower_quantity_revision_of_filled_command_requires_review(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                original = self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '2')], order_status='filled')
                with self.assertRaisesRegex(RecoveryBlocked, 'filled_quantity_correction'):
                    self.run_reconcile(values, fills=[fill(values[3], values[2], 'f', '1', version=2)])
                self.assertEqual(values[1].load(values[3].scope_id, values[3].id), original)
                self.assertEqual(values[0].command_for_intent(values[2]).state, CommandState.FILLED)
                self.assertEqual(values[0].projection(values[3].scope_id).reserved_cash, 1)
                self.assertTrue(values[0].recovery_attention(values[3].scope_id))

    def test_repeated_receive_time_is_idempotent_but_economic_conflict_is_not(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                store, ledger, intent, command, claim = self.pair(backend)
                event = fill(command,intent,'f','1')
                original = self.run_reconcile((store,ledger,intent,command,claim), fills=[event])
                result = reconcile_lifecycle(store,ledger,command=command,intent=intent,claim=claim,
                    venue_order_id='venue-order-1', fills=[replace(event,observed_at=NOW+timedelta(seconds=4))],
                    observed_at=NOW+timedelta(seconds=4), clock=lambda: NOW+timedelta(seconds=4))
                self.assertEqual(result, original)
                with self.assertRaisesRegex(RecoveryBlocked, 'conflicting_fill_id_revision'):
                    self.run_reconcile((store,ledger,intent,command,claim), fills=[replace(event,quantity=Decimal('2'))])
                self.assertEqual(ledger.load(command.scope_id,command.id), original)
                self.assertTrue(store.recovery_attention(command.scope_id))

    def test_late_fill_after_terminal_release_is_preserved_for_operator(self):
        for backend in self.backends:
            for status in ('cancelled', 'rejected'):
                with self.subTest(backend=backend, status=status):
                    values = self.pair(backend, acknowledge=status != 'rejected')
                    store, ledger, intent, command, _ = values
                    self.run_reconcile(values, order_status=status, terminal_evidence_complete=True)
                    original = ledger.load(command.scope_id, command.id)
                    replacement = store.claim_command(command.id, worker_id='late-fill-reader',
                        now=NOW+timedelta(seconds=33), lease_seconds=30)
                    self.assertIsNotNone(replacement)
                    with self.assertRaisesRegex(RecoveryBlocked, 'terminal_fill_evidence_changed'):
                        self.run_reconcile((store, ledger, intent, command, replacement),
                            at=NOW+timedelta(seconds=34), fills=[fill(command, intent, 'late', '1')])
                    self.assertEqual(ledger.load(command.scope_id, command.id), original)
                    self.assertEqual(store.projection(command.scope_id).reserved_cash, 0)
                    self.assertEqual(store.recovery_attention(command.scope_id)[0]['evidence']['fills'][0]['fill_id'], 'late')
                    with self.assertRaisesRegex(TwinStoreError, 'paused'):
                        store.begin_dispatch(command.id, worker_id=replacement.worker_id,
                            fence=replacement.fence, now=NOW+timedelta(seconds=34))

    def test_older_status_cannot_replace_newer_open_evidence(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                store, ledger, intent, command, claim = self.pair(backend)
                first = reconcile_lifecycle(store, ledger, command=command, intent=intent, claim=claim,
                    venue_order_id='venue-order-1', order_status='open', observed_at=NOW+timedelta(seconds=10),
                    clock=lambda: NOW+timedelta(seconds=10))
                result = reconcile_lifecycle(store, ledger, command=command, intent=intent, claim=claim,
                    venue_order_id='venue-order-1', order_status='rejected', terminal_evidence_complete=True,
                    observed_at=NOW+timedelta(seconds=5), clock=lambda: NOW+timedelta(seconds=11))
                self.assertEqual(result, first)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 1)
                self.assertEqual(store.recovery_attention(command.scope_id), ())

    def test_wrong_command_claim_and_slow_lookup_cannot_commit(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend, acknowledge=False)
                store, _, intent, command, claim = values
                with self.assertRaises(RecoveryBlocked):
                    recover_submission(store, command=command, intent=intent, claim=replace(claim,command_id='other'),
                        now=NOW+timedelta(seconds=3), lookups=[lookup(command,intent,found=True)],
                        clock=lambda: NOW+timedelta(seconds=3))
                now = [NOW+timedelta(seconds=3)]
                def delayed(_now=now, _command=command, _intent=intent):
                    yield lookup(_command, _intent, found=True)
                    _now[0] = NOW+timedelta(seconds=33)
                with self.assertRaises(RecoveryBlocked):
                    recover_submission(store, command=command, intent=intent, claim=claim,
                        now=NOW+timedelta(seconds=3), lookups=delayed(), clock=lambda _now=now: _now[0])
                self.assertEqual(store.command_for_intent(intent).state, CommandState.SUBMISSION_UNKNOWN)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 1)

    def test_wrong_epoch_absence_cannot_release_cash_but_current_epoch_can(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                store, _, intent, command, claim = self.pair(backend, acknowledge=False)
                observations = [lookup(command,intent,found=False,offset=offset) for offset in (3,8,13)]
                result = recover_submission(store, command=command, intent=intent, claim=claim,
                    now=NOW+timedelta(seconds=13), lookups=[replace(row,account_epoch=2) for row in observations],
                    clock=lambda: NOW+timedelta(seconds=13))
                self.assertEqual(result.action, recovery.RecoveryAction.OPERATOR_ATTENTION)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 1)
                result = recover_submission(store, command=command, intent=intent, claim=claim,
                    now=NOW+timedelta(seconds=13), lookups=observations,
                    clock=lambda: NOW+timedelta(seconds=13))
                self.assertEqual(result.action, recovery.RecoveryAction.CONFIRMED_ABSENT)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 0)

    def test_cancel_response_after_lease_expiry_is_not_accepted(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                store, _, intent, command, claim = self.pair(backend)
                now, calls = [NOW+timedelta(seconds=3)], []
                def cancel(current, _now=now, _calls=calls):
                    _calls.append(current.client_order_id)
                    _now[0] = NOW+timedelta(seconds=33)
                    return {'status':'cancelled'}
                with self.assertRaises(RecoveryBlocked):
                    cancel_after_reconciliation(store, command=command, intent=intent, claim=claim,
                        now=NOW+timedelta(seconds=3), lookup=lookup(command,intent,found=True),
                        clock=lambda _now=now: _now[0], cancel=cancel)
                self.assertEqual(calls, [command.client_order_id])
                self.assertEqual(store.command_for_intent(intent).state, CommandState.CANCEL_REQUESTED)
                self.assertEqual(store.projection(command.scope_id).reserved_cash, 1)

    def test_failed_transition_does_not_emit_a_committed_fill_metric(self):
        for backend in self.backends:
            with self.subTest(backend=backend):
                values = self.pair(backend)
                with mock.patch.object(recovery.recovery_operations, 'add') as metric:
                    with mock.patch.object(values[0], 'transition_command', side_effect=TwinStoreError('injected')):
                        with self.assertRaises(RecoveryBlocked):
                            self.run_reconcile(values, fills=[fill(values[3],values[2],'f','1')])
                self.assertEqual([call.args[1]['state'] for call in metric.call_args_list], ['blocked'])
                self.assert_held_without_evidence(values)


class AccountEpochAndSQLiteTests(unittest.TestCase):
    def test_interruption_during_transaction_entry_restores_depth(self):
        class Cancelled(BaseException):
            pass
        class InterruptClient(Client):
            interrupt = False
            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if name == '_tx_depth' and value == 1 and self.interrupt:
                    self.interrupt = False
                    raise Cancelled()
        client = InterruptClient(':memory:')
        self.addCleanup(client.close)
        client.interrupt = True
        with self.assertRaises(Cancelled):
            with client.transaction():
                self.fail('body must not run after interrupted entry')
        self.assertEqual(client._tx_depth, 0)
        self.assertFalse(client._conn.in_transaction)
        with client.transaction():
            pass

    def test_snapshot_epoch_is_explicit_and_legacy_rows_remain_unbound(self):
        case = atomic.AtomicLifecycleTests(methodName='runTest')
        self.addCleanup(case.doCleanups)
        _, _, intent, command, _ = case.pair('memory')
        current = account_snapshot(command)
        client = Client(':memory:')
        self.addCleanup(client.close)
        for repo in (InMemoryAccountSnapshotStore(), DatastoreAccountSnapshotStore(client)):
            repo.save(current)
            restored = repo.load(command.scope_id)
            self.assertEqual(restored.account_epoch, 1)
            self.assertTrue(recovery.lookup_from_complete_account(restored, command=command,
                intent=intent, observed_at=NOW).complete)
            for epoch in (None, 2, True):
                with self.assertRaises(RecoveryBlocked):
                    recovery.lookup_from_complete_account(replace(restored,account_epoch=epoch),
                        command=command, intent=intent, observed_at=NOW)
            repo.save(replace(current, generation=2, account_epoch=None))
            legacy = repo.load(command.scope_id)
            self.assertIsNone(legacy.account_epoch)
            with self.assertRaises(RecoveryBlocked):
                recovery.lookup_from_complete_account(legacy, command=command, intent=intent, observed_at=NOW)

    def test_commit_failure_rolls_back_and_resets_sqlite_transaction_depth(self):
        client = Client(':memory:')
        self.addCleanup(client.close)
        item = Entity(key=client.key('Fixture', 'rollback'))
        item['value'] = 'uncommitted'
        with mock.patch.object(client, '_conn', wraps=client._conn) as conn:
            conn.commit.side_effect = sqlite3.OperationalError('injected commit fault')
            with self.assertRaises(sqlite3.OperationalError):
                with client.transaction():
                    client.put(item)
        self.assertEqual(client._tx_depth, 0)
        self.assertFalse(client._conn.in_transaction)
        self.assertIsNone(client.get(item.key))
        with client.transaction():
            client.put(item)
        self.assertEqual(client.get(item.key)['value'], 'uncommitted')

    def test_lifecycle_commit_failure_is_a_typed_block_with_no_partial_commit(self):
        case = atomic.AtomicLifecycleTests(methodName='runTest')
        self.addCleanup(case.doCleanups)
        values = case.pair('sql')
        with mock.patch.object(values[0]._client, '_conn', wraps=values[0]._client._conn) as conn:
            conn.commit.side_effect = sqlite3.OperationalError('injected backend failure')
            with self.assertRaisesRegex(RecoveryBlocked, 'commit unconfirmed'):
                case.run_reconcile(values, fills=[fill(values[3],values[2],'f','1')])
        case.assert_held_without_evidence(values)
