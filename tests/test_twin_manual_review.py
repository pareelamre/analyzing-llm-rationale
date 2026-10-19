"""Review regressions for native manual authority, migration and cleanup."""
import importlib.util
import os
import socket
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta
from decimal import Decimal
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from analyzing_llm_rationale.twin import polymarket_manual as manual
from analyzing_llm_rationale.twin.execution import ExecutionBlocked
from analyzing_llm_rationale.twin.manual import (
    _digest,
    owner_scope_ref,
    reserve_confirmed_manual_order,
)
from analyzing_llm_rationale.twin.models import AccountScope, CommandState, TradeIntent
from tests import test_twin_native_capture as fixtures

NOW = fixtures.NOW


@unittest.skipUnless(importlib.util.find_spec('py_clob_client_v2'), 'optional trading SDK absent')
class ManualReviewTests(unittest.TestCase):
    def setUp(self):
        self.real_connect = socket.socket.connect
        self.real_addrinfo = socket.getaddrinfo
        self.f = fixtures.NativeManualTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        patcher = mock.patch.dict(os.environ, {'FORESEA_ENABLE_BYO_TRADING':'true'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.credentials = {'fixture':'scoped'}
        notify = mock.patch('analyzing_llm_rationale.server._notify_trading_safety_event')
        notify.start()
        self.addCleanup(notify.stop)

    def local_event_loop(self):
        # Windows implements asyncio's self-pipe using a loopback socketpair.
        # Keep all non-loopback connections and DNS denied during ASGI tests.
        def connect(sock, address):
            if not isinstance(address, tuple) or address[0] not in {'127.0.0.1','::1'}:
                raise AssertionError('external network denied')
            return self.real_connect(sock, address)

        def addrinfo(host, *args, **kwargs):
            if host not in {'127.0.0.1','::1'}:
                raise AssertionError('external DNS denied')
            return self.real_addrinfo(host, *args, **kwargs)

        stack = ExitStack()
        stack.enter_context(mock.patch.object(socket.socket, 'connect', new=connect))
        stack.enter_context(mock.patch('socket.getaddrinfo', new=addrinfo))
        return stack

    def http_client(self, path, handler):
        # Isolate the production HTTP handler from the unrelated singleton MCP
        # lifespan, which cannot be started twice in one test process.
        app = FastAPI()
        app.add_api_route(path, handler, methods=['POST'])
        return TestClient(app)

    def submit(self, claim, *, policy=None, disconnected=False):
        self.f.trade = claim.intent
        with mock.patch.object(manual, 'manual_polymarket_client', return_value=self.f.client), \
             mock.patch.object(manual, 'capture_manual_polymarket',
                return_value=(self.f.raw_market,self.f.raw_books,NOW)):
            return manual.submit_confirmed_polymarket_order(self.f.store, command_claim=claim,
                payload=self.f.manual_payload, user_id='manual-user', credentials=self.credentials,
                confirmation='PLACE REAL ORDER', clock=lambda:NOW,
                load_credentials=lambda:None if disconnected else self.credentials,
                load_policy=lambda:policy or {'paused':False,'max_daily_risk_notional':20})

    def test_default_gate_is_disabled_without_an_environment_override(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ExecutionBlocked, 'not enabled'):
            manual.require_native_manual_enabled()

    def test_pause_disconnect_and_flag_off_release_only_unstarted_work(self):
        for index, fault in enumerate(('pause','disconnect','flag')):
            with self.subTest(fault=fault):
                claim = self.f.prepare_manual(request_id='blocked-'+str(index))
                with mock.patch.dict(os.environ, {'FORESEA_ENABLE_NATIVE_MANUAL_TRADING':
                    'false' if fault == 'flag' else 'true'}), self.assertRaises(ExecutionBlocked):
                    self.submit(claim, policy={'paused':True,'max_daily_risk_notional':20}
                        if fault == 'pause' else None, disconnected=fault == 'disconnect')
                self.assertEqual(self.f.store.projection(claim.scope.id).reserved_cash, 0)
                self.assertEqual(self.f.store.command_for_intent(claim.intent).state, CommandState.REJECTED)
        self.assertEqual(self.f.post.call_count, 0)

    def test_cleanup_cannot_release_an_already_started_sender(self):
        claim = self.f.prepare_manual()
        self.assertTrue(self.f.store.begin_dispatch(claim.command.id, worker_id=claim.worker_id,
                                                    fence=claim.fence, now=NOW))
        with self.assertRaisesRegex(ExecutionBlocked, 'policy'):
            self.submit(claim, policy={'paused':True,'max_daily_risk_notional':20})
        self.assertGreater(self.f.store.projection(claim.scope.id).reserved_cash, 0)
        self.assertEqual(self.f.post.call_count, 0)

    def test_external_cash_reduction_is_rechecked_before_transport(self):
        claim = self.f.prepare_manual()
        self.f.available_units = '1000000'  # One USD, not one million USD.
        with self.assertRaisesRegex(ExecutionBlocked, 'cash'):
            self.submit(claim)
        projection = self.f.store.projection(claim.scope.id)
        self.assertEqual(projection.venue_available_cash, Decimal('1'))
        self.assertEqual(projection.reserved_cash, 0)
        self.assertEqual(self.f.post.call_count, 0)

    def test_native_resting_order_is_rejected_until_lifecycle_is_integrated(self):
        with self.assertRaisesRegex(ExecutionBlocked, 'FAK/FOK only'):
            self.f.prepare_manual(normalized_updates={'exchange_order':{'order_type':'GTC','post_only':False}})
        self.assertEqual(self.f.post.call_count, 0)

    def test_confirmed_exchange_rejection_returns_an_auditable_rejected_result(self):
        from analyzing_llm_rationale import server

        claim = self.f.prepare_manual()
        self.f.post.side_effect = None
        self.f.post.return_value = {'success':False,'errorMsg':'fixture-invalid','orderID':''}
        result = self.submit(claim)
        audit = server._submitted_trading_order(result)
        self.assertFalse(result['submitted'])
        self.assertEqual(audit['status'], 'rejected')
        self.assertEqual(audit['venue_order_id'], claim.command.client_order_id)
        self.assertEqual(self.f.store.projection(claim.scope.id).reserved_cash, 0)

    def legacy(self, *, index=0, terminal=False, action='buy'):
        claim = reserve_confirmed_manual_order(self.f.store, user_id='manual-user', venue='polymarket',
            payload={'client_order_id':'legacy-'+str(index)}, normalized={'action':action,'outcome':'yes',
                'token_id':'123','price':'.1','quantity':'1','estimated_notional':'.1'},
            guardrails={'policy':{'max_daily_risk_notional':20},'portfolio':{'available':100}},
            authority_ref='legacy-'+str(index), now=NOW)
        if terminal:
            self.f.store.transition_command(claim.command.id, target=CommandState.REJECTED,
                worker_id=claim.worker_id, fence=claim.fence)
        return claim

    def test_strong_legacy_key_blocks_even_when_global_inventory_misses_it(self):
        old = self.legacy(action='sell')  # Zero cash does not prove absence of work.
        with mock.patch.object(self.f.store, 'account_scopes', return_value=()), \
             self.assertRaisesRegex(ExecutionBlocked, 'previous manual account'):
            self.f.prepare_manual()
        self.assertTrue(self.f.store.has_unresolved_work(old.scope.id))
        self.assertEqual(self.f.post.call_count, 0)

    def test_terminal_history_larger_than_the_old_page_limit_does_not_lock_cutover(self):
        for index in range(205):
            old = self.legacy(index=index, terminal=True)
        self.assertFalse(self.f.store.has_unresolved_work(old.scope.id))
        claim = self.f.prepare_manual()
        self.assertNotEqual(claim.scope.id, old.scope.id)
        self.assertTrue(claim.command.client_order_id.startswith('0x'))

    def test_another_native_maker_is_not_misclassified_as_legacy_work(self):
        owner, maker = owner_scope_ref('manual-user'), '0x'+'22'*20
        other = AccountScope('account-'+_digest(owner,'polymarket',maker,'live','USD')[:32],
            owner,'polymarket',maker,'live','USD',
            'connection-'+_digest('connection','manual-user','polymarket')[:32],1,NOW)
        self.f.store.register_account(other, venue_available_cash=Decimal('10'), loss_limit=Decimal('10'))
        trade = TradeIntent('other-wallet-work',other.id,1,'polymarket:live:other:market',
            'BUY_YES',Decimal('1'),Decimal('.1'),'IOC',None,'manual-other', 'manual-guardrails-v1',
            'manual-confirmed-v1','quote',Decimal('0'),Decimal('0'),NOW.replace(year=2027),NOW)
        self.f.store.reserve_intent(trade,cash=Decimal('1'),max_loss=Decimal('1'),now=NOW)
        claim = self.f.prepare_manual()
        self.assertNotEqual(claim.scope.id, other.id)
        self.assertEqual(self.f.store.projection(other.id).reserved_cash, Decimal('1'))

    def test_external_authority_and_client_alias_each_prevent_cross_flow_rebinding(self):
        claim = self.f.prepare_manual(request_id='client-1', authority_override='saved-run-1')
        for request, authority in (('client-2','saved-run-1'),('client-1','saved-run-2')):
            with self.subTest(request=request,authority=authority), self.assertRaisesRegex(ExecutionBlocked,'persistence'):
                self.f.prepare_manual(request_id=request, authority_override=authority,
                                      when=NOW+timedelta(seconds=2))
        self.assertEqual(len(self.f.store.commands(claim.scope.id)),1)

    def test_direct_http_requires_a_stable_client_id_before_any_order_can_be_sent(self):
        from analyzing_llm_rationale import server

        payload = {'platform':'polymarket','market_id':'67890','token_id':'123', 'order_type':'limit',
            'time_in_force':'FAK','quantity':10,'price':.42,'execute':True,'confirmation':'PLACE REAL ORDER'}
        with mock.patch.object(server, '_require_session', return_value={'sub':'manual-user'}), \
             mock.patch.object(server, '_check_rate_limit'), \
             mock.patch.object(server, '_stored_trading_credentials', return_value=self.credentials), \
             mock.patch.object(server, '_record_trading_risk_event'), self.local_event_loop(), \
             self.http_client('/trading/orders', server.trading_order) as client:
            for _ in range(2):
                response = client.post('/trading/orders', json=payload)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIn('client_order_id', str(response.json()['detail']))
            response = client.post('/trading/orders', json={**payload,
                'execute':False,'client_order_id':'explicitly-not-executing'})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn('execute=true', str(response.json()['detail']))
        self.assertEqual(self.f.post.call_count, 0)

    def test_disabled_saved_run_is_not_claimed_or_consumed(self):
        from analyzing_llm_rationale import server

        run = {'id':'run-1','platform':'polymarket','status':'awaiting_approval',
            'order_request':{'platform':'polymarket','token_id':'123','market_id':'67890',
                'quantity':10,'price':.42,'time_in_force':'FAK','client_order_id':'saved-request'}}
        with mock.patch.object(server, '_require_session', return_value={'sub':'manual-user'}), \
             mock.patch.object(server, '_check_rate_limit'), \
             mock.patch.object(server, '_read_trading_run', return_value=run), \
             mock.patch.object(server, '_claim_trading_run_for_execution') as claim, \
             mock.patch.object(server, '_put_trading_run') as put, \
             mock.patch.object(server, '_stored_trading_credentials', return_value=self.credentials), \
             mock.patch.object(server, '_record_trading_risk_event'), \
             mock.patch.dict(os.environ, {'FORESEA_ENABLE_NATIVE_MANUAL_TRADING':'false'}), \
             self.local_event_loop(), \
             self.http_client('/trading/runs/{run_id}/execute', server.execute_trading_run) as client:
            response = client.post('/trading/runs/run-1/execute', json={'confirmation':'PLACE REAL ORDER'})
        self.assertEqual(response.status_code, 409, response.text)
        claim.assert_not_called()
        put.assert_not_called()
        self.assertEqual(run['status'], 'awaiting_approval')

    def test_direct_http_records_a_confirmed_venue_rejection(self):
        from analyzing_llm_rationale import market_data, server, trading

        self.f.capture()
        self.f.raw_market['feesEnabled'] = False
        self.f.post.side_effect = None
        self.f.post.return_value = {'success':False,'errorMsg':'fixture-invalid','orderID':''}
        payload = {'platform':'polymarket','market_id':'67890','token_id':'123', 'order_type':'limit',
            'time_in_force':'FAK','quantity':10,'price':.42,'execute':True,
            'confirmation':'PLACE REAL ORDER','client_order_id':'rejected-http-request'}
        guardrails = {'policy':{'max_daily_risk_notional':20},'portfolio':{'available':100}}

        def public_read(url, params=None, **kwargs):
            if url == market_data.POLYMARKET_GAMMA_URL:
                return [self.f.raw_market]
            if url == market_data.POLYMARKET_CLOB_BOOK_URL:
                return self.f.raw_books[params['token_id']]
            raise AssertionError('unexpected public data read')

        with mock.patch.object(server, '_require_session', return_value={'sub':'manual-user'}), \
             mock.patch.object(server, '_check_rate_limit'), \
             mock.patch.object(server, 'datetime', wraps=datetime) as time, \
             mock.patch.object(server, '_stored_trading_credentials', return_value=self.credentials), \
             mock.patch.object(server, '_confirmed_manual_twin_store', return_value=self.f.store), \
             mock.patch.object(server, '_validate_live_trade_guardrails', new=mock.AsyncMock(return_value=guardrails)), \
             mock.patch.object(server, '_effective_trading_guardrails',
                return_value={'paused':False,'max_daily_risk_notional':20}), \
             mock.patch.object(server, '_put_trading_order', side_effect=lambda _, row:row) as put, \
             mock.patch.object(server, '_record_terminal_trading_order_event') as terminal, \
             mock.patch.object(server, '_record_trading_risk_event'), \
             mock.patch.object(trading, '_polymarket_client', return_value=self.f.client), \
             mock.patch.object(market_data, '_get_json', side_effect=public_read), self.local_event_loop(), \
             self.http_client('/trading/orders', server.trading_order) as client:
            time.now.return_value = NOW
            response = client.post('/trading/orders', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['submitted'])
        self.assertEqual(put.call_args.args[1]['status'], 'rejected')
        terminal.assert_called_once()
        self.assertEqual(self.f.post.call_count, 1)

    def test_public_capture_failure_does_not_consume_a_saved_run(self):
        from analyzing_llm_rationale import market_data, server

        run = {'id':'run-2','platform':'polymarket','status':'awaiting_approval',
            'order_request':{'platform':'polymarket','token_id':'123','market_id':'67890',
                'quantity':10,'price':.42,'time_in_force':'FAK','client_order_id':'saved-request-2'}}
        guardrails = {'policy':{'max_daily_risk_notional':20},'portfolio':{'available':100}}
        with mock.patch.object(server, '_require_session', return_value={'sub':'manual-user'}), \
             mock.patch.object(server, '_check_rate_limit'), \
             mock.patch.object(server, '_read_trading_run', return_value=run), \
             mock.patch.object(server, '_claim_trading_run_for_execution') as claim, \
             mock.patch.object(server, '_put_trading_run') as put, \
             mock.patch.object(server, '_stored_trading_credentials', return_value=self.credentials), \
             mock.patch.object(server, '_validate_live_trade_guardrails', new=mock.AsyncMock(return_value=guardrails)), \
             mock.patch.object(server, '_record_trading_risk_event'), \
             mock.patch.object(server, '_notify_trading_safety_event'), \
             mock.patch.object(manual, 'capture_manual_polymarket',
                side_effect=market_data.MarketDataError('fixture public data unavailable')) as capture, \
             self.local_event_loop(), \
             self.http_client('/trading/runs/{run_id}/execute', server.execute_trading_run) as client:
            response = client.post('/trading/runs/run-2/execute', json={'confirmation':'PLACE REAL ORDER'})
        self.assertEqual(response.status_code, 500, response.text)
        capture.assert_called_once()
        claim.assert_not_called()
        put.assert_not_called()
        self.assertEqual(run['status'], 'awaiting_approval')
