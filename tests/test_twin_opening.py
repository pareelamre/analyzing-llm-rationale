"""Owner-approved initial baselines use real durable state, without venue calls."""
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.account import synchronize_account
from analyzing_llm_rationale.twin.account_journal import (
    DatastoreAccountJournal,
    JournalFact,
    JournalOpening,
    JournalPosition,
    compare_journal_snapshot,
)
from analyzing_llm_rationale.twin.account_store import DatastoreAccountSnapshotStore
from analyzing_llm_rationale.twin.models import AccountScope, TradeIntent
from analyzing_llm_rationale.twin.opening import AccountOpeningService, create_opening_router
from analyzing_llm_rationale.twin.store import DatastoreTwinStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
EVIDENCE = 'e' * 64  # synthetic digest, not an account or credential


class AccountOpeningTests(unittest.TestCase):
    venue = 'kalshi'

    def make_client(self):
        return self.enterContext(closing(Client(path=str(Path(self.root) / 'state.sqlite3'))))

    def setUp(self):
        self.root = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.dict('os.environ', {'FORESEA_DATASTORE_BACKEND': 'sql'}))
        self.client = self.make_client()
        self.store = DatastoreTwinStore(self.client)
        self.snapshots = DatastoreAccountSnapshotStore(self.client)
        self.scope = AccountScope('opening-account', 'owner', self.venue, 'fixture-account',
                                  'live', 'USD', 'fixture-connection', 1, NOW)
        self.store.register_account(self.scope, venue_available_cash=D('2'), loss_limit=D('1'))
        self.time = NOW + timedelta(seconds=4)
        self.snapshot = self.make_snapshot()
        self.snapshots.save(self.snapshot)
        self.service = AccountOpeningService(self.store, self.snapshots, clock=lambda: self.time)
        app = FastAPI()
        app.include_router(create_opening_router(lambda: self.service,
            resolve_owner=lambda request: request.headers.get('X-Fixture-Owner', '')))
        self.http = self.enterContext(TestClient(app))
        self.url = f'/twin/accounts/{self.scope.id}/opening'
        self.headers = {'X-Fixture-Owner': 'owner'}

    def make_snapshot(self, *, generation=1, orders=(), fills=(), positions=None):
        def page(items):
            return {'complete': True, 'items': list(items)}
        result = synchronize_account(self.scope.id, generation=generation, account_epoch=1,
            received_at=NOW + timedelta(seconds=generation),
            balances=[page([{'available': '10', 'total': '10', 'reserved': '0', 'settled_cash': '10'}])],
            positions=[page(positions if positions is not None else [
                {'position_id': 'native-token', 'quantity': '2', 'basis': '.8', 'liquidation_value': '.6'}])],
            orders=[page(orders)], fills=[page(fills)], settlements=[page(())], local_command_ids=set())
        self.assertIsNotNone(result.snapshot)
        return result.snapshot

    def review(self):
        response = self.http.get(self.url, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def body(self, review=None):
        return {'review_hash': (review or self.review())['review_hash'], 'evidence_ref': EVIDENCE,
                'confirmation': 'REGISTER ACCOUNT OPENING'}

    def confirm(self, body=None):
        return self.http.post(self.url, headers=self.headers, json=body or self.body())

    def test_owner_registration_persists_baseline_and_approval_without_capital(self):
        before = self.store.projection(self.scope.id)
        review = self.review()
        self.assertEqual((review['cash'], review['fees_paid'], review['positions'][0]['basis']),
                         ('10', '0', '0.8'))
        response = self.confirm(self.body(review))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['status'], 'registered')
        journal = DatastoreAccountJournal(self.store).load(self.scope)
        self.assertEqual((journal.opening.cash, journal.opening.positions[0].quantity), (D('10'), D('2')))
        after = self.store.projection(self.scope.id)
        self.assertEqual(replace(after, revision=before.revision), before)
        self.assertEqual(after.revision, before.revision + 1)
        self.assertTrue(self.snapshots.load(self.scope.id).blocks_new_exposure)
        self.assertEqual(self.review(), response.json())

    def test_review_is_read_only_and_never_initializes_automatically(self):
        before = self.store.projection(self.scope.id)
        self.review()
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))
        self.assertEqual(self.store.projection(self.scope.id), before)

    def test_review_exposes_the_identity_and_snapshot_behind_its_hashes(self):
        import json
        from hashlib import sha256

        review = self.review()
        self.assertEqual(review['scope'], self.scope.to_storage())
        self.assertEqual(review['snapshot']['scope_id'], self.scope.id)
        for field, reference in (('scope', 'scope_ref'), ('snapshot', 'snapshot_ref')):
            raw = json.dumps(review[field], sort_keys=True, separators=(',', ':'))
            self.assertEqual(sha256(raw.encode()).hexdigest(), review[reference])

    def test_adopted_history_reconciles_but_changed_or_added_rows_block(self):
        orders = ({'order_id': 'old-order', 'client_order_id': 'prior-manual', 'status': 'filled'},)
        fills = ({'fill_id': 'old-fill', 'order_id': 'old-order', 'fee': '0', 'status': 'CONFIRMED'},)
        self.snapshots.save(self.make_snapshot(generation=2, orders=orders, fills=fills))
        self.assertEqual(self.confirm().status_code, 200)
        journal = DatastoreAccountJournal(self.store).load(self.scope)
        unchanged = self.make_snapshot(generation=3, orders=orders, fills=fills)
        self.assertTrue(unchanged.blocks_new_exposure)
        reconciled = compare_journal_snapshot(unchanged, journal)
        self.assertFalse(reconciled.blocks_new_exposure)
        self.snapshots.save(reconciled)
        self.assertFalse(self.snapshots.load(self.scope.id).blocks_new_exposure)
        for changed in (({**fills[0], 'price': '.123'},),
                        (*fills, {**fills[0], 'fill_id': 'new-fill'})):
            current = self.make_snapshot(generation=4, orders=orders, fills=changed)
            result = compare_journal_snapshot(current, journal)
            self.assertTrue(result.blocks_new_exposure)
            self.assertEqual(result.external_activity_ids, ('prior-manual',))

    def test_zero_quantity_basis_must_be_zero_and_review_matches_opening(self):
        self.snapshots.save(self.make_snapshot(generation=2, positions=(
            {'position_id': 'closed', 'quantity': '0', 'basis': '.8', 'liquidation_value': '0'},)))
        self.assertEqual(self.http.get(self.url, headers=self.headers).status_code, 409)
        self.snapshots.save(self.make_snapshot(generation=3, positions=(
            {'position_id': 'closed', 'quantity': '0', 'basis': '0', 'liquidation_value': '0'},)))
        review = self.review()
        self.assertEqual(review['positions'], [])
        self.assertEqual(self.confirm(self.body(review)).status_code, 200)
        self.assertEqual(DatastoreAccountJournal(self.store).load(self.scope).opening.positions, ())

    def test_anonymous_and_other_owner_cannot_read_or_confirm(self):
        body = self.body()
        for owner in ('', 'other'):
            for method in ('get', 'post'):
                with self.subTest(owner=owner, method=method):
                    response = getattr(self.http, method)(self.url,
                        headers={'X-Fixture-Owner': owner}, **({'json': body} if method == 'post' else {}))
                    self.assertEqual(response.status_code, 403)
                    self.assertNotIn('native-token', response.text)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))
        absent = self.http.get('/twin/accounts/missing/opening', headers=self.headers)
        foreign = self.http.get(self.url, headers={'X-Fixture-Owner': 'other'})
        self.assertEqual((absent.status_code, absent.json()), (foreign.status_code, foreign.json()))

    def test_session_resolver_http_rejection_is_preserved_before_storage(self):
        factory = Mock()
        def reject(_request):
            raise HTTPException(401, 'Sign in required.')
        app = FastAPI()
        app.include_router(create_opening_router(factory, resolve_owner=reject))
        with TestClient(app) as client:
            self.assertEqual(client.get(self.url).status_code, 401)
            self.assertEqual(client.post(self.url, json=self.body()).status_code, 401)
        factory.assert_not_called()

    def test_actual_server_owner_resolver_and_coherent_store_wiring(self):
        from analyzing_llm_rationale import server
        from analyzing_llm_rationale.twin.manual import owner_scope_ref

        owner = owner_scope_ref('synthetic-session-subject')
        root = self.client.get(self.store._key(self.scope.id))
        root['owner_id'] = owner
        self.client.put(root)
        class FixedDate(datetime):
            @classmethod
            def now(cls, tz=None):
                return NOW + timedelta(seconds=4)
        with (patch.object(server, '_manual_twin_store', self.store),
              patch.object(server, 'datetime', FixedDate),
              patch.object(server, '_require_session', return_value={'sub': 'synthetic-session-subject'}),
              closing(TestClient(server.app)) as client):
            review = client.get(self.url)
            self.assertEqual(review.status_code, 200, review.text)
            response = client.post(self.url, json=self.body(review.json()))
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['status'], 'registered')

    def test_changed_snapshot_or_root_revision_rejects_review(self):
        body = self.body()
        self.snapshots.save(self.make_snapshot(generation=2))
        self.assertEqual(self.confirm(body).status_code, 409)
        body = self.body()
        self.store.refresh_account_capacity(self.scope.id, venue_available_cash=D('1'), loss_limit=D('1'))
        self.assertEqual(self.confirm(body).status_code, 409)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))

    def test_local_command_and_held_money_prevent_adoption(self):
        body = self.body()
        trade = TradeIntent('first', self.scope.id, 1, f'{self.venue}:live:native-token', 'BUY_YES',
            D('1'), D('.4'), 'IOC', 'forecast', None, 'policy', 'strategy', 'market',
            D('.01'), D('.01'), NOW + timedelta(minutes=5), NOW)
        self.store.reserve_intent(trade, cash=D('.5'), max_loss=D('.5'), now=NOW)
        before = self.store.projection(self.scope.id)
        self.assertEqual(self.confirm(body).status_code, 409)
        self.assertEqual(self.store.projection(self.scope.id), before)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))

    def test_connection_metadata_is_bound_to_the_owner_review(self):
        body = self.body()
        root = self.client.get(self.store._key(self.scope.id))
        root['connection_ref'] = 'changed-connection'
        self.client.put(root)
        self.assertEqual(self.confirm(body).status_code, 409)

    def test_registered_approval_cannot_be_replayed_after_scope_corruption(self):
        body = self.body()
        self.assertEqual(self.confirm(body).status_code, 200)
        root = self.client.get(self.store._key(self.scope.id))
        root['connection_ref'] = 'corrupt-same-epoch-reference'
        self.client.put(root)
        self.assertEqual(self.confirm(body).status_code, 503)

    def test_registered_approval_checks_the_actual_opening_economics(self):
        body = self.body()
        self.assertEqual(self.confirm(body).status_code, 200)
        journals = DatastoreAccountJournal(self.store)
        current = journals.load(self.scope)
        # Simulate a corrupted trusted storage writer; hashes are not MACs.
        with self.client.transaction():
            journals._write(replace(current, opening=replace(current.opening, cash=D('11'))))
        self.assertEqual(self.confirm(body).status_code, 503)

    def test_decimal_scale_does_not_invalidate_a_registered_opening(self):
        snapshot = self.make_snapshot(generation=2)
        snapshot = replace(snapshot, total_cash=D('10.0000'), available_cash=D('10.0000'),
            settled_cash=D('10.0000'), fees_paid=D('0.0000'),
            holdings=tuple(replace(row, quantity=D('2.0000'), basis=D('.8000')) for row in snapshot.holdings))
        self.snapshots.save(snapshot)
        first = self.confirm()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(self.review(), first.json())

    def test_stale_or_future_snapshot_rejects_confirmation(self):
        body = self.body()
        for time in (NOW, NOW + timedelta(minutes=3)):
            self.time = time
            self.assertEqual(self.confirm(body).status_code, 409)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))

    def test_reconnect_does_not_accept_old_epoch_snapshot(self):
        body = self.body()
        root = self.client.get(self.store._key(self.scope.id))
        root['account_epoch'] = 2
        self.client.put(root)
        self.assertEqual(self.confirm(body).status_code, 409)

    def test_active_or_unknown_orders_cannot_be_adopted(self):
        for index, status in enumerate(('resting', 'LIVE', None, 'NEW_STATUS'), start=2):
            snapshot = self.make_snapshot(generation=index, orders=({'order_id': 'venue-order', 'status': status},))
            self.snapshots.save(snapshot)
            self.assertEqual(self.http.get(self.url, headers=self.headers).status_code, 409)

    def test_unconfirmed_polymarket_fill_blocks_opening(self):
        if self.venue != 'polymarket':
            return
        self.snapshots.save(self.make_snapshot(generation=2,
            fills=({'fill_id': 'trade', 'fee': '0', 'status': 'MINED'},)))
        self.assertEqual(self.http.get(self.url, headers=self.headers).status_code, 409)

    def test_terminal_failed_polymarket_history_can_be_adopted_without_a_fee(self):
        if self.venue != 'polymarket':
            return
        self.snapshots.save(self.make_snapshot(generation=2,
            fills=({'fill_id': 'failed-trade', 'fee': '0', 'status': 'FAILED'},)))
        self.assertEqual(self.confirm().status_code, 200)

    def test_duplicate_restart_and_conflicting_confirmation_preserve_opening(self):
        body = self.body()
        first = self.confirm(body)
        self.assertEqual(first.status_code, 200, first.text)
        self.time += timedelta(hours=2)
        other = self.make_client()
        self.service = AccountOpeningService(DatastoreTwinStore(other), DatastoreAccountSnapshotStore(other),
                                             clock=lambda: self.time)
        self.assertEqual(self.confirm(body).json(), first.json())
        self.assertEqual(self.confirm({**body, 'evidence_ref': 'f' * 64}).status_code, 409)
        self.assertEqual(len(DatastoreAccountJournal(self.store).load(self.scope).facts), 0)

    def test_retry_does_not_reset_later_journal_facts(self):
        body = self.body()
        first = self.confirm(body).json()
        self.time += timedelta(seconds=2)
        journals = DatastoreAccountJournal(self.store, clock=lambda: self.time)
        current = journals.load(self.scope)
        current = journals.append(self.scope, (JournalFact('fixture-buy', 1, 'fill', self.time,
            D('-.2'), (JournalPosition('native-token', D('1'), D('.2')),), D('0'), EVIDENCE),),
            expected_revision=current.revision)
        self.assertEqual(self.confirm(body).json(), first)
        self.assertEqual(journals.load(self.scope), current)

    def test_existing_unattested_opening_cannot_be_overwritten_or_approved(self):
        body = self.body()
        journals = DatastoreAccountJournal(self.store, clock=lambda: self.time)
        prior = journals.initialize(JournalOpening(self.scope, NOW, D('0'), (), D('0'), EVIDENCE))
        self.assertEqual(self.confirm(body).status_code, 409)
        self.assertEqual(journals.load(self.scope), prior)

    def test_corrupt_approval_does_not_trigger_a_repair_or_reinitialize(self):
        body = self.body()
        self.assertEqual(self.confirm(body).status_code, 200)
        key = self.service._key(self.scope)
        entity = self.client.get(key)
        entity['fingerprint'] = '0' * 64
        self.client.put(entity)
        prior = DatastoreAccountJournal(self.store).load(self.scope)
        self.assertEqual(self.confirm(body).status_code, 503)
        self.assertEqual(DatastoreAccountJournal(self.store).load(self.scope), prior)

    def test_deadline_expiring_during_initialization_rolls_back_every_write(self):
        body = self.body()
        before = self.store.projection(self.scope.id)
        original = self.service.journals.initialize
        def expire_after_initialize(*args, **kwargs):
            result = original(*args, **kwargs)
            self.time += timedelta(minutes=3)
            return result
        with patch.object(self.service.journals, 'initialize', side_effect=expire_after_initialize):
            self.assertEqual(self.confirm(body).status_code, 409)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))
        self.assertIsNone(self.client.get(self.service._key(self.scope)))
        self.assertEqual(self.store.projection(self.scope.id), before)

    def test_explicit_confirmation_and_strict_bounded_body_are_required(self):
        body = self.body()
        for changes in ({'confirmation': 'yes'}, {'owner_id': 'owner'}, {'evidence_ref': 'not-a-digest'},
                        {'review_hash': 'x' * 64}, {'cash': '1000000'}):
            self.assertEqual(self.confirm({**body, **changes}).status_code, 422)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))

    def test_lost_approval_write_rolls_back_journal_and_root(self):
        body = self.body()
        before = self.store.projection(self.scope.id)
        original = self.client.put
        def fail_approval(entity):
            if entity.key.kind == 'TwinAccountOpeningApproval':
                raise RuntimeError('private fixture storage failure')
            return original(entity)
        with patch.object(self.client, 'put', side_effect=fail_approval):
            response = self.confirm(body)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private fixture', response.text)
        self.assertIsNone(DatastoreAccountJournal(self.store).load(self.scope))
        self.assertEqual(self.store.projection(self.scope.id), before)


class PolymarketAccountOpeningTests(AccountOpeningTests):
    venue = 'polymarket'
