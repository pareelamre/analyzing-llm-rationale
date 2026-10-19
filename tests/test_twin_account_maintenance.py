"""Native account maintenance through the existing private worker operation."""
import json
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.account_maintenance import (
    AccountReadBinding,
    NativeAccountMaintenance,
)
from analyzing_llm_rationale.twin.account_store import (
    AccountSnapshotStoreError,
    DatastoreAccountSnapshotStore,
)
from analyzing_llm_rationale.twin.runtime_app import _maintenance_operation
from analyzing_llm_rationale.twin.store import DatastoreTwinStore
from analyzing_llm_rationale.twin.venue_account import (
    complete_account_read_plan,
    conservative_bid_liquidation,
)
from analyzing_llm_rationale.twin.worker import (
    DatastoreWorkerJobs,
    WorkerDegraded,
    WorkerJob,
    WorkerJobKind,
)
from tests.test_twin_recovery import NOW, prepared_unknown


class NativeAccountMaintenanceTests(unittest.TestCase):
    venue = 'kalshi'

    def setUp(self):
        self.root = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.dict('os.environ', {'FORESEA_DATASTORE_BACKEND': 'sql'}))
        self.client = self.make_client()
        self.store, self.intent, self.command, _ = prepared_unknown(
            venue=self.venue, environment='live', store=DatastoreTwinStore(self.client),
        )
        self.scope = self.store.account_scope(self.command.scope_id)
        self.credentials = ({'kalshi_api_key_id':'fixture-key-a', 'kalshi_private_key':'fixture-not-a-real-key'}
            if self.venue=='kalshi' else {
                'polymarket_api_key':'fixture-key-a', 'polymarket_private_key':'fixture-not-a-real-key',
                'polymarket_api_secret':'fixture-not-a-real-secret', 'polymarket_api_passphrase':'fixture-passphrase'})
        account_ref = ('kalshi-key-'+sha256(b'fixture-key-a').hexdigest()
                       if self.venue=='kalshi' else '0x'+'a'*40)
        account = self.client.get(self.store._key(self.scope.id))
        account['venue_account_ref'] = account_ref
        self.client.put(account)
        self.scope = self.store.account_scope(self.scope.id)
        def client_for_credentials(creds):
            address = '0x'+('b' if creds['polymarket_api_key'].endswith('-b') else 'a')*40
            return SimpleNamespace(host='https://clob.polymarket.com', builder=SimpleNamespace(funder=address),
                                   signer=SimpleNamespace(address=lambda:address))
        self.enterContext(patch('analyzing_llm_rationale.trading._polymarket_client', side_effect=client_for_credentials))
        self.snapshots = DatastoreAccountSnapshotStore(self.client)
        self.jobs = DatastoreWorkerJobs(self.client)
        job = WorkerJob('native-reconcile', self.scope.id, WorkerJobKind.RECONCILE,
                        {'account_epoch_id':'1'}, NOW+timedelta(minutes=3), created_at=NOW)
        self.jobs.add(job)
        self.job = self.jobs.claim(job.id, worker_id='maintenance', now=NOW, lease_seconds=120)
        self.time = NOW+timedelta(seconds=3)
        fixture_name = 'kalshi_account_v2.json' if self.venue=='kalshi' else 'polymarket_account_v1.json'
        self.fixture = json.loads((Path(__file__).parent/'fixtures/twin'/fixture_name).read_text())
        self.fixture['orders']['orders' if self.venue=='kalshi' else 'data'][0]['client_order_id'] = self.command.client_order_id
        self.hook = lambda: None
        self.balance_reads = 0

    def make_client(self):
        return self.enterContext(closing(Client(path=str(Path(self.root)/'state.sqlite3'))))

    def reader(self, _venue, operation, _parameters, **_kwargs):
        if operation in {'historical_orders', 'historical_fills'}:
            return {'data':{operation.removeprefix('historical_'):[]}, 'next_cursor':None}
        if operation == 'balance':
            self.balance_reads += 1
            if self.balance_reads == 2:
                self.hook()
        return {'data':json.loads(json.dumps(self.fixture[operation])), 'next_cursor':None}

    def binding(self, scope, *, credentials=None, liquidation_reader=lambda *_: Decimal('1.20')):
        def snapshot_reader():
            self.balance_reads += 1
            if self.balance_reads == 2:
                self.hook()
            return json.loads(json.dumps(self.fixture['accounting']))
        return AccountReadBinding(scope, complete_account_read_plan(
            scope.venue, reader=self.reader, creds=credentials or self.credentials,
            liquidation_reader=liquidation_reader, connection_ref=scope.connection_ref,
            accounting_snapshot_reader=snapshot_reader if self.venue=='polymarket' else None,
        ))

    def operation(self, resolver=None):
        return NativeAccountMaintenance(
            self.store, self.snapshots, self.jobs, resolve_plan=resolver or self.binding,
            clock=lambda:self.time,
        )

    def test_private_reconcile_selects_native_reader_and_persists_explicit_epoch(self):
        before = self.store.projection(self.scope.id)
        result = _maintenance_operation(self.jobs, None, self.job, account_maintenance=self.operation())
        snapshot = self.snapshots.load(self.scope.id)
        self.assertEqual(result, {'status':'complete', 'account_generation':1, 'account_epoch':1,
                                 'blocks_new_exposure':False, 'orders_observed':1, 'fills_observed':1})
        self.assertEqual(snapshot.account_epoch, 1)
        self.assertEqual(snapshot.available_cash, Decimal('9'))
        self.assertEqual(self.store.projection(self.scope.id), before)
        self.assertEqual(self.store.command_for_intent(self.intent), self.command)
        self.assertNotIn('fixture', json.dumps(result))

    def test_reconnect_during_read_cannot_publish_old_epoch(self):
        def reconnect():
            root = self.client.get(self.store._key(self.scope.id))
            root['account_epoch'] = 2
            self.client.put(root)
        self.hook = reconnect
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))
        self.assertEqual(self.store.projection(self.scope.id).account_epoch, 2)

    def test_expired_or_refenced_job_cannot_publish(self):
        def refence():
            self.jobs.claim(self.job.id, worker_id='new-worker', now=NOW+timedelta(seconds=121), lease_seconds=30)
        for hook in (lambda:setattr(self, 'time', NOW+timedelta(seconds=121)), refence):
            with self.subTest(hook=hook):
                self.time = NOW+timedelta(seconds=3)
                self.hook = hook
                self.balance_reads = 0
                with self.assertRaises(WorkerDegraded):
                    self.operation()(self.job)
                self.assertGreater(self.balance_reads, 0)
                self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_bound_plan_cannot_read_a_different_connection_or_venue(self):
        other_venue = 'polymarket' if self.venue=='kalshi' else 'kalshi'
        for field, value in (('connection_ref','other'), ('venue_account_ref','other'), ('venue',other_venue)):
            with self.subTest(field=field):
                good = self.binding(self.scope)
                def resolver(_scope, good=good, field=field, value=value):
                    return replace(good, scope=replace(self.scope, **{field:value}))
                with self.assertRaises(WorkerDegraded):
                    self.operation(resolver)(self.job)
                self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_changed_local_projection_or_command_cannot_publish(self):
        self.hook = lambda:self.store.refresh_account_capacity(
            self.scope.id, venue_available_cash=Decimal('8'), loss_limit=Decimal('10'),
        )
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_command_only_change_is_detected_without_projection_change(self):
        before = self.store.projection(self.scope.id)
        def mutate_command():
            entity = self.client.get(self.store._key(self.scope.id, 'TwinCommand', self.command.id))
            entity['dispatch_started'] = True
            self.client.put(entity)
        self.hook = mutate_command
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(self.store.projection(self.scope.id), before)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_wrong_credentials_with_correct_scope_echo_are_refused(self):
        bad = dict(self.credentials)
        bad['kalshi_api_key_id' if self.venue=='kalshi' else 'polymarket_api_key'] = 'fixture-key-b'
        def resolver(scope):
            return self.binding(scope, credentials=bad)
        with self.assertRaisesRegex(WorkerDegraded, 'native_account_reader_binding_changed'):
            self.operation(resolver)(self.job)
        self.assertEqual(self.balance_reads, 0)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_market_liquidation_changes_do_not_look_like_account_mutations(self):
        values = iter((Decimal('.1'), Decimal('.2'), Decimal('.3')))
        def resolver(scope):
            return self.binding(scope, liquidation_reader=lambda *_:next(values))
        result = self.operation(resolver)(self.job)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.snapshots.load(self.scope.id).holdings[0].liquidation_value, Decimal('.2'))

    def test_reordered_identical_rows_do_not_look_like_account_mutations(self):
        key = 'orders' if self.venue=='kalshi' else 'data'
        self.fixture['orders'][key].append({**self.fixture['orders'][key][0], 'order_id':'external', 'id':'external'})
        original, reads = self.reader, 0
        def reader(*args, **kwargs):
            nonlocal reads
            value = original(*args, **kwargs)
            if args[1]=='orders':
                reads += 1
                if reads%2==0:
                    value['data'][key].reverse()
            return value
        self.reader = reader
        self.assertEqual(self.operation()(self.job)['status'], 'complete')

    def test_capture_does_not_query_command_history_for_every_page(self):
        with patch.object(self.store, 'commands', wraps=self.store.commands) as query:
            self.operation()(self.job)
        self.assertLessEqual(query.call_count, 4)

    def test_default_orderbook_reads_cannot_start_after_deadline(self):
        calls = 0
        def book(*_):
            nonlocal calls
            calls += 1
            self.time += timedelta(seconds=91)
            return {'yes_dollars':[['.5','2']], 'bids':[['.5','2']]}
        name = 'kalshi' if self.venue=='kalshi' else 'polymarket'
        with patch('analyzing_llm_rationale.market_data.fetch_'+name+'_orderbook', side_effect=book):
            with self.assertRaises(WorkerDegraded):
                self.operation(lambda scope:self.binding(scope, liquidation_reader=None))(self.job)
        self.assertEqual(calls, 1)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_oversized_snapshot_retains_previous_generation(self):
        self.operation()(self.job)
        prior = self.snapshots.load(self.scope.id)
        tiny = DatastoreAccountSnapshotStore(self.client, max_payload_bytes=1)
        with self.assertRaises(AccountSnapshotStoreError):
            tiny.save(replace(prior, generation=2))
        self.assertEqual(self.snapshots.load(self.scope.id), prior)

    def test_failed_read_retains_prior_and_sanitizes_exception(self):
        operation = self.operation()
        operation(self.job)
        prior = self.snapshots.load(self.scope.id)
        def failed(_scope):
            raise ValueError('private credential text must never escape')
        with self.assertRaises(WorkerDegraded) as error:
            self.operation(failed)(self.job)
        self.assertNotIn('private credential', str(error.exception))
        self.assertEqual(self.snapshots.load(self.scope.id), prior)

    def test_mid_read_only_economic_revision_is_not_a_complete_generation(self):
        original = self.reader
        reads = 0
        def reader(venue, operation, params, **kwargs):
            nonlocal reads
            result = original(venue, operation, params, **kwargs)
            if operation == 'fills':
                reads += 1
                key = 'fills' if self.venue=='kalshi' else 'data'
                field = 'count_fp' if self.venue=='kalshi' else 'size'
                result['data'][key][0][field] = '9.00' if reads == 2 else '1.00'
            return result
        self.reader = reader
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_snapshot_guard_failure_after_write_rolls_back(self):
        self.operation()(self.job)
        prior = self.snapshots.load(self.scope.id)
        called = 0
        def guard():
            nonlocal called
            called += 1
            if called == 2:
                raise RuntimeError('lease expired before commit')
        with self.assertRaises(RuntimeError):
            self.snapshots.save(replace(prior, generation=2), validate=guard)
        self.assertEqual(self.snapshots.load(self.scope.id), prior)

    def test_truncated_command_inventory_never_certifies_complete_local_history(self):
        with patch.object(self.store, 'commands', return_value=(self.command,)*200):
            with self.assertRaises(WorkerDegraded):
                self.operation()(self.job)
        self.assertEqual(self.balance_reads, 0)
        self.assertIsNone(self.snapshots.load(self.scope.id))

    def test_capture_deadline_stops_before_another_page_request(self):
        original = self.reader
        calls = 0
        def reader(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = original(*args, **kwargs)
            self.time += timedelta(seconds=91)
            return result
        self.reader = reader
        with self.assertRaises(WorkerDegraded):
            self.operation()(self.job)
        self.assertEqual(calls, 1)
        self.assertIsNone(self.snapshots.load(self.scope.id))


class PolymarketAccountMaintenanceTests(NativeAccountMaintenanceTests):
    venue = 'polymarket'

    def test_native_rows_without_invented_client_ids_report_drift(self):
        self.fixture['orders']['data'][0].pop('client_order_id')
        result = self.operation()(self.job)
        self.assertTrue(result['blocks_new_exposure'])
        self.assertIn('unattributed_order:order-1', self.snapshots.load(self.scope.id).external_activity_ids)

    def test_implicit_shared_funder_cannot_replace_scoped_signer(self):
        client = SimpleNamespace(host='https://clob.polymarket.com',
            builder=SimpleNamespace(funder='0x'+'a'*40), signer=SimpleNamespace(address=lambda:'0x'+'b'*40))
        with patch('analyzing_llm_rationale.trading._polymarket_client', return_value=client):
            with self.assertRaises(WorkerDegraded):
                self.operation()(self.job)
        self.assertEqual(self.balance_reads, 0)
        self.assertIsNone(self.snapshots.load(self.scope.id))


class NativeEconomicEvidenceTests(unittest.TestCase):
    def test_liquidation_uses_explicit_price_units_and_rejects_out_of_range(self):
        from analyzing_llm_rationale.twin.reconcile import AccountReadError

        self.assertEqual(conservative_bid_liquidation({'bids':[['50','2']]}, Decimal('2'), price_unit='cents'), Decimal('1'))
        self.assertEqual(conservative_bid_liquidation({'bids':[['.50','2']]}, Decimal('2')), Decimal('1'))
        for value in ('1.0001', '200', 'NaN'):
            with self.subTest(value=value), self.assertRaises(AccountReadError):
                conservative_bid_liquidation({'bids':[[value,'2']]}, Decimal('2'))
    def test_native_fill_economics_survive_existing_page_adapters(self):
        for venue, raw, wanted in (
            ('kalshi', {'fill_id':'f', 'order_id':'o', 'count_fp':'1.25', 'fee_cost':'0',
                        'ticker':'KXTEST', 'created_time':'2025-01-01T00:00:00Z',
                        'yes_price_dollars':'0.4', 'outcome_side':'yes'}, 'count_fp'),
            ('polymarket', {'id':'f', 'taker_order_id':'o', 'size':'1.25', 'fee':'0',
                            'asset_id':'token', 'match_time':'1735689600', 'status':'CONFIRMED',
                            'price':'0.4', 'maker_orders':[{'order_id':'maker','matched_amount':'0.2'}]}, 'size'),
        ):
            with self.subTest(venue=venue):
                plan = complete_account_read_plan(venue, reader=lambda *_a, raw=raw, venue=venue, **_kw: {
                    'data':{'fills':[raw]} if venue=='kalshi' else {'data':[raw]}, 'next_cursor':None,
                }, creds={'fixture':'not-a-real-credential'})
                result = plan.fetchers['fills'](None)['items'][0]
                self.assertEqual(result[wanted], '1.25')
                self.assertEqual(result['fee'], '0')
                self.assertIn('source_fingerprint', result)
