"""Real normalized market -> native signer -> durable command integration."""
import importlib.util
import json
import unittest
from dataclasses import replace
from datetime import timedelta

from analyzing_llm_rationale.twin.execution import SubmissionDisposition
from analyzing_llm_rationale.twin.market import normalize_market
from tests import test_twin_native_polymarket as native_fixture

NOW = native_fixture.NOW


@unittest.skipUnless(importlib.util.find_spec('py_clob_client_v2'), 'optional trading SDK absent')
class NativeCaptureTests(unittest.TestCase):
    setUp = native_fixture.NativePolymarketTests.setUp
    metadata = native_fixture.NativePolymarketTests.metadata
    ack = native_fixture.NativePolymarketTests.ack
    reserve = native_fixture.NativePolymarketTests.reserve
    send = native_fixture.NativePolymarketTests.send

    def capture(self):
        market = {'id': '67890', 'conditionId': 'fixture-condition',
            'outcomes': ['Yes', 'No'], 'clobTokenIds': ['123', '456'],
            'active': True, 'closed': False, 'acceptingOrders': True,
            'endDate': (NOW+timedelta(days=1)).isoformat(), 'updatedAt': NOW.isoformat(),
            'rules': 'Resolve against the published official result.', 'category': 'sports',
            'minimum_tick_size': '.01', 'minimum_order_size': '5'}
        books = {token: {**self.book, 'asset_id': token,
            'bids': [{'price': '.41', 'size': '100'}],
            'asks': [{'price': '.42', 'size': '100'}], 'timestamp': NOW.isoformat()}
            for token in ('123','456')}
        captured = normalize_market('polymarket', market, received_at=NOW,
            sequence=1, orderbooks=books)
        self.assertTrue(captured.eligible)
        self.instrument, self.snapshot = captured.instrument, captured.snapshot
        self.raw_market, self.raw_books = market, books
        self.trade = replace(self.trade, instrument_id=self.instrument.id,
                             market_version=self.snapshot.id)
        self.context = replace(self.context, market_version=self.snapshot.id)

    def test_actual_capture_preserves_distinct_gamma_and_condition_identity(self):
        self.capture()
        self.assertEqual(self.instrument.id, 'polymarket:live:fixture-condition:67890')
        self.assertEqual(self.instrument.capability_version, 'polymarket-clob-limit')
        command = self.reserve()
        self.assertEqual(json.loads(command.prepared_request_json)['venue_instrument_id'], '67890')
        result = self.send(command)
        self.assertEqual(result.disposition, SubmissionDisposition.ACKNOWLEDGED)
        self.assertEqual(self.post.call_count, 1)


@unittest.skipUnless(importlib.util.find_spec('py_clob_client_v2'), 'optional trading SDK absent')
class NativeManualTests(NativeCaptureTests):
    def setUp(self):
        from unittest import mock

        super().setUp()
        self.available_units = '100000000'
        patcher = mock.patch.dict('os.environ', {'FORESEA_ENABLE_NATIVE_MANUAL_TRADING':'true',
                                                'FORESEA_ALLOW_MARKET_ORDERS':'true'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def metadata(self, endpoint, headers=None, params=None):
        if endpoint.endswith('/balance-allowance'):
            self.assertEqual(params['asset_type'], 'COLLATERAL')
            return {'balance':self.available_units}
        return super().metadata(endpoint, headers=headers, params=params)

    def test_actual_capture_preserves_distinct_gamma_and_condition_identity(self):
        # Keep the upstream capture contract covered here without relying on a
        # second live/native fixture or a different signer implementation.
        super().test_actual_capture_preserves_distinct_gamma_and_condition_identity()

    def prepare_manual(self, *, when=NOW, confirmation='PLACE REAL ORDER', request_id='manual-request-1',
                       market_updates=None, normalized_updates=None, payload_updates=None, authority_override=None):
        from analyzing_llm_rationale.twin.polymarket_manual import (
            prepare_confirmed_polymarket_order,
        )

        self.capture()
        self.raw_market['feesEnabled'] = False
        self.raw_market.update(market_updates or {})
        normalized = {'platform':'polymarket','action':'buy','outcome':'yes',
                'quantity':10,'price':.42,'token_id':'123','order_type':'limit',
                'exchange_order':{'order_type':'FAK','post_only':False}}
        normalized.update(normalized_updates or {})
        self.manual_payload = {'platform':'polymarket','market_id':'67890','token_id':'123',
            'action':'buy','outcome':'yes','order_type':'limit','time_in_force':'FAK',
            'quantity':10,'price':.42,'client_order_id':request_id, **(payload_updates or {})}
        return prepare_confirmed_polymarket_order(self.store, user_id='manual-user',
            payload=self.manual_payload, confirmation=confirmation, normalized=normalized,
            guardrails={'policy':{'max_daily_risk_notional':20},'portfolio':{'available':100}},
            credentials={'fixture':'scoped'}, client=self.client, market=self.raw_market,
            books=self.raw_books, authority_ref=authority_override or 'direct-'+request_id, now=when, clock=lambda:when)

    def test_manual_confirmation_and_runtime_flags_are_required_before_signing(self):
        from unittest import mock

        from analyzing_llm_rationale.trading import TradingDisabledError
        from analyzing_llm_rationale.twin.execution import ExecutionBlocked

        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}), mock.patch.object(
            self.client, 'create_order', wraps=self.client.create_order,
        ) as sign:
            with self.assertRaises(ExecutionBlocked):
                self.prepare_manual(confirmation='yes')
            with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'false'}), self.assertRaises(TradingDisabledError):
                self.prepare_manual()
            with mock.patch.dict('os.environ', {'FORESEA_ENABLE_NATIVE_MANUAL_TRADING':'false'}), self.assertRaises(ExecutionBlocked):
                self.prepare_manual()
            self.assertEqual(sign.call_count, 0)
        self.assertEqual(self.post.call_count, 0)

    def test_manual_immediate_order_has_native_hash_without_autonomous_mandate(self):
        from unittest import mock

        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            claim = self.prepare_manual()
        packet = json.loads(claim.command.prepared_request_json)
        self.assertEqual(packet['order_type'], 'FAK')
        self.assertEqual(packet['authority_mode'], 'manual')
        self.assertEqual(claim.scope.venue_account_ref.lower(), self.client.builder.funder.lower())
        self.assertTrue(claim.command.client_order_id.startswith('0x'))
        self.assertEqual(self.post.call_count, 0)

    def test_retry_later_cannot_allocate_a_second_native_manual_request(self):
        from unittest import mock

        from analyzing_llm_rationale.twin.execution import ExecutionBlocked

        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            first = self.prepare_manual()
            before = self.store.projection(first.scope.id).reserved_cash
            with self.assertRaisesRegex(ExecutionBlocked, 'persistence'):
                self.prepare_manual(when=NOW+timedelta(seconds=2))
        self.assertEqual(self.store.projection(first.scope.id).reserved_cash, before)
        self.assertEqual(len(self.store.commands(first.scope.id)), 1)

    def test_fee_capture_is_required_and_cash_includes_conservative_fee_bound(self):
        from decimal import Decimal
        from unittest import mock

        from analyzing_llm_rationale.twin.execution import ExecutionBlocked

        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            with self.assertRaises(ExecutionBlocked):
                self.prepare_manual(market_updates={'feesEnabled':None})
            claim = self.prepare_manual(market_updates={'feesEnabled':True,
                'feeSchedule':{'rate':'.02','exponent':'1','takerOnly':True}})
        self.assertEqual(self.store.projection(claim.scope.id).reserved_cash, Decimal('4.25'))
        self.assertEqual(claim.intent.fee_allowance, Decimal('.05'))

    def test_max_cost_and_authoritative_outcome_cannot_be_overridden(self):
        from unittest import mock

        from analyzing_llm_rationale.twin.execution import ExecutionBlocked

        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            for kwargs in ({'payload_updates':{'max_cost':4.1}},
                           {'normalized_updates':{'token_id':'456'}}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ExecutionBlocked):
                    self.prepare_manual(**kwargs)
        self.assertEqual(self.post.call_count, 0)

    def test_manual_lost_ack_retains_hold_and_cannot_repost(self):
        from unittest import mock

        from analyzing_llm_rationale import trading
        from analyzing_llm_rationale.twin import polymarket_manual as manual
        from analyzing_llm_rationale.twin.execution import SubmissionUnknown

        credentials = {'fixture':'scoped'}
        with mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            claim = self.prepare_manual()
            self.trade = claim.intent
            self.post.side_effect = TimeoutError('fixture-secret-must-not-leak')
            with mock.patch.object(manual, 'manual_polymarket_client', return_value=self.client), \
                 mock.patch.object(manual, 'capture_manual_polymarket', return_value=(self.raw_market,self.raw_books,NOW)):
                args = dict(command_claim=claim, payload=self.manual_payload, user_id='manual-user',
                    credentials=credentials, confirmation='PLACE REAL ORDER', clock=lambda:NOW,
                    load_credentials=lambda:credentials,
                    load_policy=lambda:{'paused':False,'max_daily_risk_notional':20})
                with self.assertRaises(SubmissionUnknown):
                    manual.submit_confirmed_polymarket_order(self.store, **args)
                with self.assertRaises(trading.TradingExecutionError):
                    manual.submit_confirmed_polymarket_order(self.store, **args)
        self.assertEqual(self.post.call_count, 1)
        self.assertGreater(self.store.projection(claim.scope.id).reserved_cash, 0)

    def test_server_manual_boundary_posts_prepared_body_without_legacy_adapter(self):
        from datetime import datetime
        from unittest import mock

        from analyzing_llm_rationale import market_data, server, trading

        self.capture()
        self.raw_market['feesEnabled'] = False
        payload = {'platform':'polymarket','market_id':'67890','token_id':'123',
            'action':'buy','outcome':'yes','order_type':'limit','time_in_force':'FAK',
            'quantity':10,'price':.42,'client_order_id':'server-native-manual'}
        credentials = {'fixture':'scoped'}

        def market_read(url, params=None, **kwargs):
            if url == market_data.POLYMARKET_GAMMA_URL:
                return [self.raw_market]
            if url == market_data.POLYMARKET_CLOB_BOOK_URL:
                return self.raw_books[params['token_id']]
            raise AssertionError('unexpected public read')

        service = mock.Mock()
        service.resolve_credentials.return_value = credentials
        with mock.patch.object(server, '_confirmed_manual_twin_store', return_value=self.store), \
             mock.patch.object(server, 'datetime', wraps=datetime) as time, \
             mock.patch.object(trading, '_polymarket_client', return_value=self.client), \
             mock.patch.object(market_data, '_get_json', side_effect=market_read), \
             mock.patch.object(server, '_effective_trading_guardrails', return_value={
                 'paused':False,'max_daily_risk_notional':20}), \
             mock.patch.dict('os.environ', {'FORESEA_ENABLE_BYO_TRADING':'true'}):
            time.now.return_value = NOW
            preview = trading.preview_order(payload, credentials)
            claim = server._reserve_confirmed_manual_order(user_id='manual-user', venue='polymarket',
                payload=payload, preview=preview, guardrails={'policy':{'max_daily_risk_notional':20},
                    'portfolio':{'available':100}}, authority_ref='direct-server-native-manual',
                credentials=credentials, confirmation='PLACE REAL ORDER')
            self.trade = claim.intent
            result = server._submit_confirmed_manual_command(claim, execution_service=service,
                payload=payload, user_id='manual-user', credentials=credentials,
                confirmation='PLACE REAL ORDER')
            audit = server._submitted_trading_order(result)
        self.assertEqual(self.post.call_count, 1)
        service.submit.assert_not_called()
        self.assertEqual(audit['venue_order_id'], claim.command.client_order_id)
        self.assertEqual(audit['status'], 'submitted')
        self.assertIsNone(audit['filled_quantity'])
