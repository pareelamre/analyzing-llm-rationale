import base64
import importlib.util
import json
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from unittest import mock

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.execution import (
    ExecutionBlocked,
    SubmissionDisposition,
    SubmissionUnknown,
    submit_claimed_command,
)
from analyzing_llm_rationale.twin.models import (
    Completeness,
    Instrument,
    MarketSnapshot,
    canonical_instrument_id,
)
from analyzing_llm_rationale.twin.polymarket_execution import (
    cancel_polymarket_command,
    lookup_polymarket_command,
    reserve_polymarket_intent,
    submit_polymarket_command,
)
from analyzing_llm_rationale.twin.store import (
    DatastoreTwinStore,
    ReservationPreconditions,
    TwinStoreError,
)
from tests.test_twin_execution import intent as base_intent
from tests.test_twin_execution import reserved

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)
TEST_ENCRYPTION_KEY = base64.urlsafe_b64encode(b'0'*32).decode()


class PreparedRequestStoreTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict('os.environ', {'FORESEA_TWIN_ORDER_ENCRYPTION_KEY': TEST_ENCRYPTION_KEY}).start()

    def test_lost_process_after_dispatch_marker_cannot_send_after_lease_takeover(self):
        store, _, trade, command, claim, context = reserved()
        self.assertTrue(store.begin_dispatch(command.id, worker_id=claim.worker_id,
                                             fence=claim.fence, now=trade.created_at))
        later = trade.created_at+timedelta(seconds=31)
        takeover = store.claim_command(command.id, worker_id='replacement', now=later)
        calls = []
        result = submit_claimed_command(store, command=command, intent=trade, claim=takeover,
                                        context=context, now=later, submit=lambda _: calls.append('sent'))
        self.assertEqual(result.disposition, SubmissionDisposition.UNKNOWN)
        self.assertEqual(calls, [])
        self.assertEqual(store.projection(context.scope.id).reserved_cash, Decimal('2.70'))

    def test_private_request_survives_sql_restart_and_rejects_mutation(self):
        with TemporaryDirectory() as root, mock.patch.dict('os.environ', {
            'FORESEA_DATASTORE_BACKEND': 'sql',
        }):
            path = str(Path(root) / 'state.sqlite3')
            _, scope, _, _, _, _ = reserved()
            trade = base_intent(scope)
            payload = {'schema_version': 1, 'signature': 'private-fixture-signature'}
            for generation in range(2):
                client = Client(path=path)
                try:
                    store = DatastoreTwinStore(client)
                    store.register_account(scope, venue_available_cash=Decimal('20'),
                                           loss_limit=Decimal('20'))
                    if generation == 0:
                        store.reserve_intent(trade, cash=Decimal('3'), max_loss=Decimal('3'),
                                             now=trade.created_at, client_order_id='native-id',
                                             prepared_request=payload)
                    command = store.command_for_intent(trade)
                    self.assertEqual(json.loads(command.prepared_request_json), payload)
                    self.assertNotIn('private-fixture-signature', repr(command))
                    with self.assertRaises(TwinStoreError):
                        store.reserve_intent(trade, cash=Decimal('3'), max_loss=Decimal('3'),
                                             client_order_id='native-id',
                                             prepared_request={**payload, 'signature': 'changed'})
                finally:
                    client.close()

    def test_bounded_plain_json_required_before_reservation(self):
        store, scope, _, _, _, _ = reserved()
        for payload in ({'bad': float('nan')}, {'bad': object()}, {'too_big': 'a'*70_000}):
            trade = replace(base_intent(scope), quantity=Decimal('1'))
            before = store.projection(scope.id)
            with self.assertRaises(TwinStoreError):
                store.reserve_intent(trade, cash=Decimal('1'), max_loss=Decimal('1'),
                                     prepared_request=payload)
            self.assertEqual(store.projection(scope.id), before)

    def test_same_claim_cannot_dispatch_twice_concurrently(self):
        store, _, trade, command, claim, context = reserved()
        started, release = Event(), Event()
        calls = []

        def post(_):
            calls.append('post')
            started.set()
            self.assertTrue(release.wait(5))
            return {'acknowledgement': {'acknowledged': True}}

        def invoke():
            return submit_claimed_command(store, command=command, intent=trade, claim=claim,
                                          context=context, now=trade.created_at, submit=post)

        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(invoke)
            self.assertTrue(started.wait(5))
            second = pool.submit(invoke)
            try:
                self.assertEqual(second.result(5).disposition, SubmissionDisposition.ALREADY_PROCESSED)
            finally:
                release.set()
            self.assertEqual(first.result(5).disposition, SubmissionDisposition.ACKNOWLEDGED)
        self.assertEqual(calls, ['post'])


@unittest.skipUnless(importlib.util.find_spec('py_clob_client_v2'), 'optional trading SDK absent')
class NativePolymarketTests(unittest.TestCase):
    def setUp(self):
        from eth_account import Account

        from analyzing_llm_rationale import trading

        self.addCleanup(mock.patch.stopall)
        mock.patch.dict('os.environ', {'FORESEA_DATASTORE_BACKEND': 'sql',
            'FORESEA_TWIN_ORDER_ENCRYPTION_KEY': TEST_ENCRYPTION_KEY}).start()
        mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('network denied')).start()
        mock.patch('socket.getaddrinfo', side_effect=AssertionError('DNS denied')).start()
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'state.sqlite3')
        self.db = Client(path=self.path)
        self.addCleanup(lambda: self.db.close())
        self.store = DatastoreTwinStore(self.db)
        _, old_scope, old_intent, _, _, old_context = reserved(environment='live')
        self.scope = replace(old_scope, venue='polymarket',
                             venue_account_ref=Account.from_key('0x'+'11'*32).address)
        canonical_id = canonical_instrument_id(
            venue='polymarket', environment='live', venue_instrument_id='fixture-condition',
            condition_id='fixture-condition',
        )
        self.trade = replace(old_intent, instrument_id=canonical_id,
                             quantity=Decimal('10'), limit_price=Decimal('.42'),
                             market_version='snapshot-native', created_at=NOW,
                             expires_at=NOW+timedelta(minutes=1))
        self.context = replace(old_context, scope=self.scope, market_version='snapshot-native',
                               mandate=replace(old_context.mandate, venue='polymarket',
                                               expires_at=NOW+timedelta(days=1)))
        self.context = replace(self.context, mandate=replace(
            self.context.mandate, approved_hash=self.context.mandate.digest(), approved_at=NOW,
        ))
        self.store.register_account(self.scope, venue_available_cash=Decimal('100'),
                                   loss_limit=Decimal('20'))
        self.instrument = Instrument(
            self.trade.instrument_id, 'polymarket', 'live', 'fixture-condition',
            'fixture-condition', '123', '456', 'rules-v1', 'sports', 'event-1', 'cluster-1',
            Decimal('.01'), Decimal('5'), 'fee-v1', 'ctf-v2', 'active',
            NOW+timedelta(days=1), NOW+timedelta(days=2), NOW-timedelta(days=1),
        )
        self.snapshot = MarketSnapshot(
            self.trade.market_version, self.instrument.id, NOW, NOW, 1, 'captured',
            Completeness.COMPLETE, 30, Decimal('.41'), Decimal('.42'), Decimal('.57'),
            Decimal('.58'), 'fee-v1', NOW,
        )
        self.book = {'asset_id': '123', 'market': 'fixture-condition', 'tick_size': '.01',
                     'min_order_size': '5', 'neg_risk': False}
        self.client = trading._polymarket_client({
            'polymarket_private_key': '0x'+'11'*32,
            'polymarket_api_key': 'fixture-api-key', 'polymarket_api_secret': 'Zml4dHVyZQ==',
            'polymarket_api_passphrase': 'fixture-passphrase',
            'polymarket_chain_id': '137', 'polymarket_clob_host': 'https://clob.polymarket.com',
            'polymarket_signature_type': '0',
        })
        mock.patch.object(self.client, '_get', side_effect=self.metadata).start()
        mock.patch('py_clob_client_v2.order_builder.builder.time.time_ns',
                   return_value=int(NOW.timestamp()*1e9)).start()
        self.post = mock.patch.object(self.client, '_post', side_effect=self.ack).start()

    def metadata(self, endpoint, headers=None, params=None):
        if endpoint.endswith('/version'):
            return {'version': 2}
        if endpoint.endswith('/tick-size'):
            return {'minimum_tick_size': '0.01'}
        raise AssertionError('unexpected SDK metadata read')

    def ack(self, endpoint, headers=None, data=None, params=None):
        # A separate SQL connection must see the native identity before transport.
        command = self.store.command_for_intent(self.trade)
        with closing(Client(path=self.path)) as separate:
            saved = DatastoreTwinStore(separate).command_for_intent(self.trade)
            self.assertEqual(saved.client_order_id, command.client_order_id)
            self.assertTrue(saved.dispatch_started)
        self.assertEqual(json.loads(data)['order'], json.loads(command.prepared_request_json)['order'])
        return {'success': True, 'orderID': command.client_order_id, 'status': 'delayed'}

    def reserve(self, **changes):
        args = dict(intent=self.trade, instrument=self.instrument, snapshot=self.snapshot,
                    book=self.book, client=self.client, context=self.context, now=NOW,
                    cash=Decimal('4.40'), max_loss=Decimal('4.40'),
                    clock=lambda: NOW,
                    preconditions=ReservationPreconditions(
                        self.store.projection(self.scope.id).revision, self.snapshot.id, NOW, 30))
        args.update(changes)
        return reserve_polymarket_intent(self.store, **args)

    def send(self, command=None, **changes):
        command = command or self.store.command_for_intent(self.trade)
        claim = self.store.claim_command(command.id, worker_id='fixture', now=NOW)
        args = dict(command=command, intent=self.trade, claim=claim, client=self.client,
                    load_context=lambda: self.context, load_instrument=lambda: self.instrument, clock=lambda: NOW)
        args.update(changes)
        return submit_polymarket_command(self.store, **args)

    def test_native_body_survives_restart_then_posts_once_without_resigning(self):
        command = self.reserve()
        self.assertTrue(command.client_order_id.startswith('0x'))
        self.assertNotIn('fixture-api-key', command.prepared_request_json)
        self.db.close()
        self.db = Client(path=self.path)
        self.store = DatastoreTwinStore(self.db)
        with mock.patch.object(self.client, 'create_order', side_effect=AssertionError('must not re-sign')):
            result = self.send(command)
            self.assertEqual(result.disposition, SubmissionDisposition.ACKNOWLEDGED)
            self.assertEqual(self.send().disposition, SubmissionDisposition.ALREADY_PROCESSED)
        self.assertEqual(self.post.call_count, 1)

    def test_simultaneous_preparations_cannot_allocate_or_send_twice(self):
        barrier = Barrier(2)
        original = self.client.create_order

        def prepare(*args, **kwargs):
            barrier.wait(5)
            return original(*args, **kwargs)

        def reserve_once():
            try:
                return self.reserve()
            except ExecutionBlocked:
                return None

        with mock.patch.object(self.client, 'create_order', side_effect=prepare), ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: reserve_once(), range(2)))
        self.assertEqual(sum(r is not None for r in results), 1)
        self.assertEqual(len(self.store.commands(self.scope.id)), 1)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))
        self.assertEqual(self.post.call_count, 0)

    def test_duplicate_during_native_restore_does_not_poison_active_sender(self):
        from analyzing_llm_rationale.twin import polymarket_execution as native

        command = self.reserve()
        claim = self.store.claim_command(command.id, worker_id='fixture', now=NOW)
        original = native._read_request
        entered, release = Event(), Event()

        def restore(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return original(*args, **kwargs)

        def invoke():
            return submit_polymarket_command(self.store, command=command, intent=self.trade,
                claim=claim, client=self.client, load_context=lambda: self.context,
                load_instrument=lambda: self.instrument, clock=lambda: NOW)

        with mock.patch.object(native, '_read_request', side_effect=restore), ThreadPoolExecutor(2) as pool:
            first = pool.submit(invoke)
            self.assertTrue(entered.wait(5))
            second = pool.submit(invoke)
            try:
                self.assertEqual(second.result(5).disposition, SubmissionDisposition.ALREADY_PROCESSED)
            finally:
                release.set()
            self.assertEqual(first.result(5).disposition, SubmissionDisposition.ACKNOWLEDGED)
        self.assertEqual(self.post.call_count, 1)

    def test_authority_refreshed_after_prepare_and_before_post(self):
        from analyzing_llm_rationale.twin.mandates import revoke

        self.reserve()
        revoked = replace(self.context, mandate=revoke(self.context.mandate,
                                                       owner_id=self.scope.owner_id))
        contexts = mock.Mock(side_effect=[self.context, revoked])
        self.assertEqual(self.send(load_context=contexts).disposition, SubmissionDisposition.REJECTED)
        self.assertEqual(contexts.call_count, 2)
        self.assertEqual(self.post.call_count, 0)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_operator_projection_and_repr_never_expose_signed_body(self):
        from analyzing_llm_rationale.twin.operator import _command_view

        command = self.reserve()
        signature = json.loads(command.prepared_request_json)['order']['signature']
        rendered = json.dumps(_command_view(command)) + repr(command)
        self.assertNotIn(signature, rendered)
        self.assertNotIn('prepared_request_json', rendered)

    def test_legacy_inflight_without_dispatch_marker_is_not_replayed(self):
        command = self.reserve()
        old = self.store.claim_command(command.id, worker_id='old', now=NOW, lease_seconds=1)
        self.assertIsNotNone(old)
        key = self.store._key(self.scope.id, 'TwinCommand', command.id)
        entity = self.db.get(key)
        entity.pop('dispatch_started')
        self.db.put(entity)
        takeover = self.store.claim_command(command.id, worker_id='new', now=NOW+timedelta(seconds=2))
        result = submit_polymarket_command(self.store, command=command, intent=self.trade,
            claim=takeover, client=self.client, load_context=lambda: self.context,
            load_instrument=lambda: self.instrument,
            clock=lambda: NOW+timedelta(seconds=2))
        self.assertEqual(result.disposition, SubmissionDisposition.UNKNOWN)
        self.assertEqual(self.post.call_count, 0)

    def test_unknown_ack_never_reposts_and_keeps_reservation(self):
        self.reserve()
        self.post.side_effect = TimeoutError('fixture-secret-must-not-leak')
        with self.assertRaises(SubmissionUnknown) as caught:
            self.send()
        self.assertNotIn('fixture-secret', str(caught.exception))
        self.assertEqual(self.send().disposition, SubmissionDisposition.ALREADY_PROCESSED)
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def test_wrong_ack_identity_is_unknown(self):
        self.reserve()
        self.post.side_effect = None
        self.post.return_value = {'success': True, 'orderID': 'different-order', 'status': 'delayed'}
        with self.assertRaises(SubmissionUnknown):
            self.send()
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def test_confirmed_rejection_releases_cash(self):
        self.reserve()
        self.post.side_effect = None
        self.post.return_value = {'success': False, 'errorMsg': 'fixture-invalid', 'orderID': ''}
        result = self.send()
        self.assertEqual(result.disposition, SubmissionDisposition.REJECTED)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_rejection_with_acceptance_evidence_is_unknown(self):
        command = self.reserve()
        self.post.side_effect = None
        self.post.return_value = {'success': False, 'errorMsg': 'contradictory',
                                  'orderID': command.client_order_id, 'status': 'matched'}
        with self.assertRaises(SubmissionUnknown):
            self.send()
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def test_rejected_flag_with_reported_fill_quantities_keeps_cash(self):
        self.reserve()
        self.post.side_effect = None
        self.post.return_value = {'success': False, 'errorMsg': 'contradictory',
                                  'makingAmount': '4.2', 'takingAmount': '10'}
        with self.assertRaises(SubmissionUnknown):
            self.send()
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def venue_order(self, command):
        wire = json.loads(command.prepared_request_json)['order']
        return {'id': command.client_order_id, 'asset_id': wire['tokenId'],
                'market': 'fixture-condition', 'maker_address': wire['maker'],
                'side': wire['side'], 'price': '.42', 'original_size': '10',
                'size_matched': '0', 'status': 'LIVE'}

    def test_native_lookup_recovers_lost_ack_after_quote_expiry_without_reposting(self):
        from analyzing_llm_rationale.twin.recovery import RecoveryAction, recover_submission

        command = self.reserve()
        self.post.side_effect = TimeoutError('lost acknowledgement')
        with self.assertRaises(SubmissionUnknown):
            self.send(command)
        later = NOW+timedelta(seconds=31)
        with mock.patch.object(self.client, '_get', return_value=self.venue_order(command)) as get, \
             mock.patch.object(self.client, 'create_order', side_effect=AssertionError('no re-sign')):
            lookup = lookup_polymarket_command(self.store, command=command, intent=self.trade,
                                                client=self.client, clock=lambda: later)
        self.assertTrue(lookup.complete)
        self.assertTrue(lookup.order_found)
        self.assertTrue(get.call_args.args[0].endswith(command.client_order_id))
        claim = self.store.claim_command(command.id, worker_id='recovery', now=later)
        result = recover_submission(self.store, command=command, intent=self.trade, claim=claim,
                                    now=later, lookups=[lookup])
        self.assertEqual(result.action, RecoveryAction.RECONCILED)
        self.assertEqual(self.post.call_count, 1)

    def test_empty_error_or_wrong_asset_lookup_never_proves_absence(self):
        command = self.reserve()
        for response in (None, {}, {**self.venue_order(command), 'asset_id': 'wrong-token'}):
            with mock.patch.object(self.client, '_get', return_value=response):
                lookup = lookup_polymarket_command(self.store, command=command, intent=self.trade,
                                                    client=self.client, clock=lambda: NOW)
            self.assertFalse(lookup.complete)
            self.assertIsNone(lookup.order_found)
        with mock.patch.object(self.client, '_get', side_effect=TimeoutError('fixture-secret')):
            lookup = lookup_polymarket_command(self.store, command=command, intent=self.trade,
                                                client=self.client, clock=lambda: NOW)
        self.assertFalse(lookup.complete)
        self.assertIsNone(lookup.order_found)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def test_native_cancel_uses_fresh_reconciliation_and_holds_cash_for_late_fills(self):
        command = self.reserve()
        self.send(command)
        claim = self.store.command_for_intent(self.trade).claim
        with mock.patch.object(self.client, '_get', return_value=self.venue_order(command)):
            lookup = lookup_polymarket_command(self.store, command=command, intent=self.trade,
                                                client=self.client, clock=lambda: NOW)
        with mock.patch.object(self.client, '_delete', return_value={
            'canceled': [command.client_order_id], 'not_canceled': {},
        }) as delete:
            result = cancel_polymarket_command(self.store, command=command, intent=self.trade,
                claim=claim, lookup=lookup, client=self.client, clock=lambda: NOW)
        self.assertTrue(result['canceled'])
        self.assertEqual(json.loads(delete.call_args.kwargs['data']), {'orderID': command.client_order_id})
        self.assertEqual(self.store.command_for_intent(self.trade).state.value, 'cancel_requested')
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('4.40'))

    def test_invalid_book_protocol_precision_or_authority_has_no_write(self):
        variants = [dict(book={**self.book, 'version': 'v2'}),
                    dict(book={**self.book, 'asset_id': '456'}),
                    dict(snapshot=replace(self.snapshot, received_at=NOW-timedelta(seconds=31),
                                          venue_at=NOW-timedelta(seconds=32))),
                    dict(intent=replace(self.trade, quantity=Decimal('5.001'))),
                    dict(context=replace(self.context, runtime_live_enabled=False))]
        for changes in variants:
            with self.subTest(changes=list(changes)), self.assertRaises(ExecutionBlocked):
                self.reserve(**changes)
        self.assertEqual(self.post.call_count, 0)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_all_binary_actions_use_the_selected_token_and_exact_side(self):
        mandate = replace(self.context.mandate,
                           allowed_actions=('BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO'))
        self.context = replace(self.context, mandate=replace(mandate, approved_hash=mandate.digest()))
        original = self.trade
        for action in ('BUY_YES', 'BUY_NO', 'SELL_YES', 'SELL_NO'):
            self.trade = replace(original, id='intent-'+action, action=action)
            token = '123' if action.endswith('_YES') else '456'
            command = self.reserve(book={**self.book, 'asset_id': token})
            wire = json.loads(command.prepared_request_json)['order']
            self.assertEqual(wire['tokenId'], token)
            self.assertEqual(wire['side'], 'BUY' if action.startswith('BUY_') else 'SELL')
            self.assertEqual(self.send(command).disposition, SubmissionDisposition.ACKNOWLEDGED)
        self.assertEqual(self.post.call_count, 4)

    def test_preparation_finishing_after_capture_expiry_has_no_reservation(self):
        with self.assertRaises(ExecutionBlocked):
            self.reserve(clock=lambda: NOW+timedelta(seconds=31))
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)
        self.assertEqual(self.post.call_count, 0)

    def test_wrong_owner_venue_maker_or_endpoint_cannot_prepare(self):
        for field, value in [('owner_id', 'wrong-owner'), ('venue', 'kalshi')]:
            mandate = replace(self.context.mandate, **{field: value})
            mandate = replace(mandate, approved_hash=mandate.digest())
            with self.assertRaises(ExecutionBlocked):
                self.reserve(context=replace(self.context, mandate=mandate))
        with mock.patch.object(self.client, 'host', 'https://offline.invalid'), self.assertRaises(ExecutionBlocked):
            self.reserve()
        with mock.patch.object(self.client.builder, 'funder', '0x'+'22'*20), self.assertRaises(ExecutionBlocked):
            self.reserve()
        self.assertEqual(self.post.call_count, 0)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_root_limit_and_cumulative_holds_cannot_exceed_mandate(self):
        with self.assertRaises(ExecutionBlocked):
            self.reserve(cash=Decimal('8'), max_loss=Decimal('4.4'))
        self.store.refresh_account_capacity(self.scope.id, venue_available_cash=Decimal('100'),
                                           loss_limit=Decimal('21'))
        with self.assertRaises(ExecutionBlocked):
            self.reserve()
        self.store.refresh_account_capacity(self.scope.id, venue_available_cash=Decimal('100'),
                                           loss_limit=Decimal('20'))
        original = self.trade
        for index in range(4):
            self.trade = replace(original, forecast_id=f'forecast-{index}')
            self.reserve()
        self.trade = replace(original, forecast_id='forecast-over-limit')
        with self.assertRaises(ExecutionBlocked):
            self.reserve()
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, Decimal('17.60'))
        self.assertEqual(self.post.call_count, 0)

    def test_changed_private_body_is_blocked_before_transport(self):
        command = self.reserve()
        key = self.store._key(self.scope.id, 'TwinCommand', command.id)
        entity = self.db.get(key)
        signature = json.loads(command.prepared_request_json)['order']['signature']
        self.assertNotIn(signature, entity['prepared_request_json'])
        self.assertTrue(entity['prepared_request_json'].startswith('fernet-v1:'))
        entity['prepared_request_json'] = entity['prepared_request_json'][:-5]+'AAAAA'
        self.db.put(entity)
        with self.assertRaises(TwinStoreError):
            self.send(command)
        self.assertEqual(self.post.call_count, 0)

    def test_stale_capture_before_transport_releases_without_venue_call(self):
        command = self.reserve()
        claim = self.store.claim_command(command.id, worker_id='fixture', now=NOW, lease_seconds=60)
        result = self.send(command, claim=claim, clock=lambda: NOW+timedelta(seconds=31))
        self.assertEqual(result.disposition, SubmissionDisposition.REJECTED)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)
        self.assertEqual(self.post.call_count, 0)

    def test_authoritative_token_mapping_is_reloaded_before_transport(self):
        self.reserve()
        result = self.send(load_instrument=lambda: replace(self.instrument, yes_token_id='999'))
        self.assertEqual(result.disposition, SubmissionDisposition.REJECTED)
        self.assertEqual(self.post.call_count, 0)
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_no_encryption_key_cannot_commit_a_private_order(self):
        with mock.patch.dict('os.environ', {}, clear=True), self.assertRaises(ExecutionBlocked):
            self.reserve()
        self.assertEqual(self.store.projection(self.scope.id).reserved_cash, 0)

    def test_exchange_domains_match_both_native_sdk_contract_configs(self):
        from py_clob_client_v2.config import get_contract_config

        from analyzing_llm_rationale.twin.polymarket_execution import _EXCHANGES

        config = get_contract_config(137)
        self.assertEqual(_EXCHANGES[False], config.exchange_v2)
        self.assertEqual(_EXCHANGES[True], config.neg_risk_exchange_v2)


if __name__ == '__main__':
    unittest.main()
