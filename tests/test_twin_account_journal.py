"""Economic journal behavior, including private native maintenance comparison."""
import unittest
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal as D
from hashlib import sha256
from unittest.mock import patch

from analyzing_llm_rationale.twin.account_journal import (
    AccountJournalError,
    DatastoreAccountJournal,
    JournalFact,
    JournalOpening,
    JournalPosition,
    replay_journal,
)
from analyzing_llm_rationale.twin.worker import WorkerDegraded
from tests import test_twin_account_maintenance as native
from tests.test_twin_recovery import NOW

REF = sha256(b'fixture verified source').hexdigest()


class AccountJournalTests(unittest.TestCase):
    """Use existing real SQL account/worker fixture, without another execution stack."""

    fixture_type = native.NativeAccountMaintenanceTests

    def setUp(self):
        self._case = self.fixture_type(methodName='runTest')
        self._case.setUp()
        self.addCleanup(self._case.doCleanups)
        self.journal = DatastoreAccountJournal(self.store, clock=lambda: self.time)

    def __getattr__(self, name):
        case = self.__dict__.get('_case')
        if case is not None and name in ('scope', 'store', 'snapshots', 'job', 'time', 'operation'):
            return getattr(case, name)
        raise AttributeError(name)

    @property
    def hook(self):
        return self._case.hook

    @hook.setter
    def hook(self, value):
        self._case.hook = value

    def opening(self, **kwargs):
        values = dict(scope=self.scope, observed_at=NOW, cash=D('10'), positions=(),
                      fees_paid=D('0'), source_ref=REF)
        values.update(kwargs)
        return JournalOpening(**values)

    def fact(self, ident='buy', **kwargs):
        values = dict(id=ident, version=1, kind='fill', occurred_at=NOW+timedelta(seconds=1),
                      cash_delta=D('-0.82'), positions=(JournalPosition('KXTEST-YES', D('2'), D('.8')),),
                      fee=D('.02'), source_ref=REF)
        values.update(kwargs)
        return JournalFact(**values)

    def test_cash_inventory_basis_fees_and_cashflows_replay(self):
        journal = self.journal.initialize(self.opening())
        journal = self.journal.append(self.scope, (self.fact(),), expected_revision=journal.revision)
        deposit = self.fact('deposit', kind='external', cash_delta=D('3'), positions=(), fee=D('0'),
                            external_cash_flow=D('3'), sequence=1)
        journal = self.journal.append(self.scope, (deposit,), expected_revision=journal.revision)
        result = replay_journal(journal, as_of=self.time)
        self.assertEqual((result.cash, result.fees_paid, result.external_cash_flow, result.realized_pnl),
                         (D('12.18'), D('.02'), D('3'), D('-.02')))
        self.assertEqual(result.positions, (JournalPosition('KXTEST-YES', D('2'), D('.8')),))

    def test_duplicate_and_corrective_settlement_do_not_credit_twice(self):
        journal = self.journal.initialize(self.opening())
        journal = self.journal.append(self.scope, (self.fact(),), expected_revision=journal.revision)
        settle = self.fact('settle', kind='settlement', cash_delta=D('2'), fee=D('0'),
                           positions=(JournalPosition('KXTEST-YES', D('-2'), D('-.8')),),
                           occurred_at=NOW+timedelta(seconds=2))
        journal = self.journal.append(self.scope, (settle,), expected_revision=journal.revision)
        duplicate = self.journal.append(self.scope, (settle,), expected_revision=journal.revision)
        self.assertEqual(duplicate, journal)
        correction = replace(settle, version=2, cash_delta=D('1'))
        corrected = self.journal.append(self.scope, (correction,), expected_revision=journal.revision)
        result = replay_journal(corrected, as_of=self.time)
        self.assertEqual((result.cash, result.positions, result.realized_pnl), (D('10.18'), (), D('.18')))
        self.assertEqual(len(corrected.facts), 3)
        with self.assertRaises(AccountJournalError):
            self.journal.append(self.scope, (replace(correction, cash_delta=D('1.1')),),
                                expected_revision=corrected.revision)

    def test_provisional_settlement_keeps_inventory_and_cash(self):
        journal = self.journal.initialize(self.opening())
        journal = self.journal.append(self.scope, (self.fact(),), expected_revision=journal.revision)
        pending = self.fact('pending', kind='settlement', final=False, cash_delta=D('2'), fee=D('0'),
                            positions=(JournalPosition('KXTEST-YES', D('-2'), D('-.8')),),
                            occurred_at=NOW+timedelta(seconds=2))
        journal = self.journal.append(self.scope, (pending,), expected_revision=journal.revision)
        result = replay_journal(journal, as_of=self.time)
        self.assertTrue(result.pending_settlement)
        self.assertEqual(result.cash, D('9.18'))
        self.assertEqual(result.positions[0].quantity, D('2'))

    def test_immutable_opening_and_revision_fencing(self):
        original = self.journal.initialize(self.opening())
        self.assertEqual(self.journal.initialize(self.opening()), original)
        with self.assertRaises(AccountJournalError):
            self.journal.initialize(self.opening(cash=D('99')))
        self.journal.append(self.scope, (self.fact(),), expected_revision=original.revision)
        with self.assertRaises(AccountJournalError):
            self.journal.append(self.scope, (), expected_revision=original.revision)
        self.assertEqual(DatastoreAccountJournal(self.store).load(self.scope), self.journal.load(self.scope))
        with self.assertRaises(AccountJournalError):
            self.journal.load(replace(self.scope, account_epoch=2))

    def test_invalid_history_cannot_balance_away_drift(self):
        original = self.journal.initialize(self.opening())
        for bad in (replace(self.fact(), version=True), replace(self.fact(), cash_delta=D('NaN')),
                    replace(self.fact(), version=2), replace(self.fact(), occurred_at=NOW-timedelta(seconds=1)),
                    replace(self.fact(), source_ref='unverified')):
            with self.subTest(bad=bad), self.assertRaises(AccountJournalError):
                self.journal.append(self.scope, (bad,), expected_revision=original.revision)
        self.assertEqual(self.journal.load(self.scope), original)

    def seed_matching_native_journal(self):
        # Explicit fixture opening. Production never bootstraps from its observation.
        instrument, basis, fees = ('KXTEST-YES', '.8', '.02') if self._case.venue == 'kalshi' else ('token-1', '1.2', '0')
        return self.journal.initialize(self.opening(
            cash=D('10'), positions=(JournalPosition(instrument, D('2'), D(basis)),),
            fees_paid=D(fees)))

    def test_native_capture_records_exact_journal_provenance(self):
        journal = self.seed_matching_native_journal()
        before = self.store.projection(self.scope.id)
        result = self.operation()(self.job)
        snapshot = self.snapshots.load(self.scope.id)
        self.assertFalse(result['blocks_new_exposure'])
        self.assertTrue(result['ledger_reconciled'])
        self.assertEqual(snapshot.journal_revision, journal.revision)
        self.assertEqual(snapshot.journal_fingerprint, journal.fingerprint)
        self.assertEqual(self.store.projection(self.scope.id), before)

    def test_native_capture_without_journal_blocks_exposure(self):
        result = self.operation()(self.job)
        self.assertTrue(result['blocks_new_exposure'])
        self.assertFalse(result['ledger_reconciled'])
        self.assertIn('journal_uninitialized', self.snapshots.load(self.scope.id).drift_reasons)

    def test_native_cash_basis_fee_and_inventory_drift_are_persisted(self):
        self.journal.initialize(self.opening(cash=D('12'), fees_paid=D('.01'),
            positions=(JournalPosition('KXTEST-YES', D('3'), D('.9')),)))
        self.assertTrue(self.operation()(self.job)['blocks_new_exposure'])
        reasons = self.snapshots.load(self.scope.id).drift_reasons
        for reason in ('journal_cash_mismatch', 'journal_basis_mismatch', 'journal_fees_mismatch',
                       'journal_holdings_mismatch'):
            self.assertIn(reason, reasons)

    def test_journal_changed_during_read_cannot_publish_stale_capture(self):
        journal = self.seed_matching_native_journal()
        self.hook = lambda: self.journal.append(self.scope, (self.fact('fee', kind='fee',
            cash_delta=D('-.01'), positions=(), fee=D('.01')),), expected_revision=journal.revision)
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_journal_changed_at_commit_cannot_publish(self):
        journal = self.seed_matching_native_journal()
        save = self.snapshots.save
        def changed(snapshot, *, validate=None):
            self.journal.append(self.scope, (self.fact('fee', kind='fee', cash_delta=D('-.01'),
                positions=(), fee=D('.01')),), expected_revision=journal.revision)
            return save(snapshot, validate=validate)
        with patch.object(self.snapshots, 'save', side_effect=changed), self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_source_proof_is_not_a_snapshot_bootstrap_or_mutable_opening(self):
        for bad in (self.opening(cash=D('-1')), self.opening(source_ref='not-a-digest'),
                    self.opening(observed_at=NOW-timedelta(days=1000)),
                    self.opening(positions=(JournalPosition('x', D('0'), D('1')),))):
            with self.subTest(bad=bad), self.assertRaises(AccountJournalError):
                self.journal.initialize(bad)
        self.assertIsNone(self.journal.load(self.scope))

    def test_replay_rejects_negative_cash_or_unexplained_external_posting(self):
        original = self.journal.initialize(self.opening())
        for bad in (replace(self.fact(), cash_delta=D('-20')),
                    replace(self.fact(), kind='external', external_cash_flow=D('-1')),
                    replace(self.fact(), cash_delta=D('1e1000000')),
                    replace(self.fact(), positions=(JournalPosition('x', D('1'), D('-1')),))):
            with self.subTest(bad=bad), self.assertRaises(AccountJournalError):
                self.journal.append(self.scope, (bad,), expected_revision=original.revision)
        self.assertEqual(self.journal.load(self.scope), original)

    def test_exact_decimal_roundtrip_and_future_read_rejection(self):
        cash = D('123456789012345678.123456789012345678')
        opening = self.journal.initialize(self.opening(cash=cash))
        self.assertEqual(self.journal.load(self.scope).opening.cash, cash)
        updated = self.journal.append(self.scope, (self.fact(),), expected_revision=opening.revision)
        with self.assertRaises(AccountJournalError):
            replay_journal(updated, as_of=NOW)

    def test_corrective_version_cannot_reorder_a_source_event(self):
        original = self.journal.initialize(self.opening())
        updated = self.journal.append(self.scope, (self.fact(),), expected_revision=original.revision)
        with self.assertRaises(AccountJournalError):
            self.journal.append(self.scope, (replace(self.fact(), version=2,
                occurred_at=NOW+timedelta(seconds=2)),), expected_revision=updated.revision)

    def test_journal_initialization_during_capture_cannot_clear_missing_authority(self):
        self.hook = self.seed_matching_native_journal
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_corrupted_history_is_not_used_as_cash_authority(self):
        journal = self.seed_matching_native_journal()
        client = self.journal._client
        entity = client.get(self.journal._key(self.scope))
        entity['payload_json'] = entity['payload_json'].replace('"cash":"10"', '"cash":"99"')
        client.put(entity)
        with self.assertRaises(AccountJournalError):
            self.journal.load(self.scope)
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))
        self.assertEqual(journal.opening.cash, D('10'))

    def test_missing_provenance_blocks_legacy_snapshot(self):
        self.seed_matching_native_journal()
        self.operation()(self.job)
        current = self.snapshots.load(self.scope.id)
        legacy = replace(current, journal_revision=None, journal_fingerprint=None)
        self.assertFalse(legacy.divergence)
        self.assertTrue(legacy.blocks_new_exposure)

    def test_committed_snapshot_is_blocked_after_journal_changes(self):
        journal = self.seed_matching_native_journal()
        self.operation()(self.job)
        self.assertFalse(self.snapshots.load(self.scope.id).blocks_new_exposure)
        self.journal.append(self.scope, (self.fact('fee', kind='fee', cash_delta=D('-.01'),
            positions=(), fee=D('.01')),), expected_revision=journal.revision)
        self.assertTrue(self.snapshots.load(self.scope.id).blocks_new_exposure)

    def test_corrupt_journal_blocks_an_existing_reconciled_snapshot(self):
        self.seed_matching_native_journal()
        self.operation()(self.job)
        key = self.journal._key(self.scope)
        entity = self.journal._client.get(key)
        entity['fingerprint'] = 'corrupted'
        self.journal._client.put(entity)
        self.assertTrue(self.snapshots.load(self.scope.id).blocks_new_exposure)

    def test_source_sequence_orders_equal_timestamp_events(self):
        journal = self.journal.initialize(self.opening())
        buy = self.fact('z-buy', sequence=0)
        settle = self.fact('a-settle', sequence=1, kind='settlement', cash_delta=D('2'), fee=D('0'),
                           positions=(JournalPosition('KXTEST-YES', D('-2'), D('-.8')),))
        journal = self.journal.append(self.scope, (buy, settle), expected_revision=journal.revision)
        self.assertEqual(replay_journal(journal, as_of=self.time).cash, D('11.18'))

    def test_voided_fill_and_strategy_income_keep_correct_economics(self):
        journal = self.journal.initialize(self.opening())
        journal = self.journal.append(self.scope, (self.fact(),), expected_revision=journal.revision)
        voided = replace(self.fact(), version=2, voided=True, cash_delta=D('0'), positions=(), fee=D('0'))
        income = self.fact('rebate', sequence=1, kind='income', cash_delta=D('.05'), positions=(), fee=D('0'))
        journal = self.journal.append(self.scope, (voided, income), expected_revision=journal.revision)
        result = replay_journal(journal, as_of=self.time)
        self.assertEqual((result.cash, result.positions, result.fees_paid, result.realized_pnl),
                         (D('10.05'), (), D('0'), D('.05')))

    def test_import_future_timestamp_is_rejected_before_storage(self):
        journal = self.seed_matching_native_journal()
        self.operation()(self.job)
        with self.assertRaisesRegex(AccountJournalError, 'journal_fact_after_import'):
            self.journal.append(self.scope, (self.fact(occurred_at=self.time+timedelta(seconds=1)),),
                                expected_revision=journal.revision)
        self.assertEqual(self.journal.load(self.scope), journal)
        self.assertFalse(self.snapshots.load(self.scope.id).blocks_new_exposure)

    def test_history_and_storage_bounds_retain_existing_journal(self):
        journal = self.seed_matching_native_journal()
        before = self.store.projection(self.scope.id)
        facts = tuple(self.fact('fee-'+str(index), sequence=index, kind='fee',
            cash_delta=D('0'), positions=(), fee=D('0')) for index in range(4097))
        with self.assertRaisesRegex(AccountJournalError, 'journal_history_limit'):
            self.journal.append(self.scope, facts, expected_revision=journal.revision)
        facts = tuple(self.fact(str(index)+'x'*248, sequence=index, kind='fee',
            cash_delta=D('0'), positions=(), fee=D('0')) for index in range(1900))
        with self.assertRaisesRegex(AccountJournalError, 'journal_storage_limit'):
            self.journal.append(self.scope, facts, expected_revision=journal.revision)
        self.assertEqual(self.journal.load(self.scope), journal)
        self.assertEqual(self.store.projection(self.scope.id), before)

    def test_legacy_roundtrip_is_readable_but_cannot_authorize_exposure(self):
        import json

        from analyzing_llm_rationale.twin.account_store import (
            AccountSnapshotStoreError,
            _canonical_json,
            _restore,
        )
        self.seed_matching_native_journal()
        self.operation()(self.job)
        current = self.snapshots.load(self.scope.id)
        legacy = replace(current, generation=2, journal_revision=None, journal_fingerprint=None,
                         account_epoch=None)
        payload = json.loads(_canonical_json(legacy))
        self.assertNotIn('journal_revision', payload)
        self.assertNotIn('journal_fingerprint', payload)
        self.assertEqual(_restore(payload), legacy)
        self.snapshots.save(legacy)
        self.assertTrue(self.snapshots.load(self.scope.id).blocks_new_exposure)
        for changes in ({'journal_revision':True, 'journal_fingerprint':REF},
                        {'journal_fingerprint':REF}, {'journal_revision':1},
                        {'journal_revision':1, 'journal_fingerprint':'corrupt'}):
            with self.subTest(changes=changes), self.assertRaises(AccountSnapshotStoreError):
                _restore({**payload, **changes})

    def test_future_fact_after_clock_regression_persists_blocking_capture(self):
        journal = self.seed_matching_native_journal()
        self.operation()(self.job)
        self.journal.append(self.scope, (self.fact('fee', kind='fee', positions=(),
            cash_delta=D('-.01'), fee=D('.01'), occurred_at=NOW+timedelta(seconds=2)),),
            expected_revision=journal.revision)
        self._case.time = NOW+timedelta(seconds=1)
        self.assertTrue(self.operation()(self.job)['blocks_new_exposure'])
        self.assertIn('journal_time_or_history_unconfirmed', self.snapshots.load(self.scope.id).drift_reasons)

    def test_journal_write_invalidates_risk_preconditions_without_changing_money(self):
        from analyzing_llm_rationale.twin.store import (
            ReservationPreconditions,
            StaleReservationPrecondition,
        )

        journal = self.seed_matching_native_journal()
        before = self.store.projection(self.scope.id)
        self.journal.append(self.scope, (self.fact('fee', kind='fee', positions=(),
            cash_delta=D('-.01'), fee=D('.01')),), expected_revision=journal.revision)
        after = self.store.projection(self.scope.id)
        self.assertEqual(replace(after, revision=before.revision), before)
        intent = replace(self._case.intent, id='new-risk-decision',
                         limit_price=self._case.intent.limit_price-D('.01'))
        preconditions = ReservationPreconditions(before.revision, intent.market_version, NOW, 60)
        with self.assertRaises(StaleReservationPrecondition):
            self.store.reserve_intent(intent, cash=D('1'), max_loss=D('1'), now=self.time,
                                      preconditions=preconditions)
        self.assertEqual(self.store.projection(self.scope.id), after)

    def test_high_precision_native_basis_is_not_rounded_to_global_context(self):
        basis = '123456789012345678.123456789012345678'
        if self._case.venue == 'kalshi':
            self._case.fixture['positions']['market_positions'][0]['market_exposure_dollars'] = basis
            instrument, fees = 'KXTEST-YES', '.02'
        else:
            self._case.fixture['accounting']['positions'][0]['initialValue'] = basis
            instrument, fees = 'token-1', '0'
        self.journal.initialize(self.opening(cash=D('10'), fees_paid=D(fees),
            positions=(JournalPosition(instrument, D('2'), D(basis)),)))
        self.assertTrue(self.operation()(self.job)['ledger_reconciled'])
        self.assertEqual(self.snapshots.load(self.scope.id).position_basis, D(basis))


class PolymarketAccountJournalTests(AccountJournalTests):
    fixture_type = native.PolymarketAccountMaintenanceTests


if __name__ == '__main__':
    unittest.main()
