"""Explicit owner attestation of the first economic baseline of a native epoch.

An attestation adopts a reviewed baseline; it does not authenticate historical
venue receipts, repair an existing journal, grant capital or activate trading.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException, Path, Request
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from .account import external_activity_refs
from .account_journal import DatastoreAccountJournal, JournalOpening, JournalPosition
from .account_store import DatastoreAccountSnapshotStore, _canonical_json
from .store import DatastoreTwinStore, _transaction_scope

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter('twin.account.opening', unit='1')


class OpeningConflict(RuntimeError):
    """The reviewed first-opening authority is no longer available."""


class OpeningConfirmation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, revalidate_instances='always')
    review_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence_ref: str = Field(pattern=r'^[0-9a-f]{64}$')
    confirmation: Literal['REGISTER ACCOUNT OPENING']


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _failed(operation):
    operations.add(1, {'operation': operation, 'outcome': 'unconfirmed'})
    trace.get_current_span().set_status(Status(StatusCode.ERROR, 'opening_unconfirmed'))
    logger.warning('Account opening requires owner reconciliation')


def _approved(scope, record, source_ref):
    return {
        'status': 'registered', 'account_scope_id': scope.id, 'account_epoch': scope.account_epoch,
        'review_hash': record['review_hash'], 'opening_ref': source_ref,
        'snapshot_generation': record['proposal']['snapshot_generation'], 'approved_at': record['approved_at'],
    }


def _attested_opening(scope, proposal, source_ref):
    return JournalOpening(scope, datetime.fromisoformat(proposal['observed_at']),
        Decimal(proposal['cash']), tuple(JournalPosition(row['instrument_id'], Decimal(row['quantity']),
                                                       Decimal(row['basis'])) for row in proposal['positions']),
        Decimal(proposal['fees_paid']), source_ref, tuple(tuple(row) for row in proposal['activity_refs']))


class AccountOpeningService:
    def __init__(self, store: DatastoreTwinStore, snapshots: DatastoreAccountSnapshotStore,
                 *, clock: Callable[[], datetime], max_age_seconds: int = 120):
        if (not isinstance(store, DatastoreTwinStore) or not isinstance(snapshots, DatastoreAccountSnapshotStore)
                or store._client is not snapshots._client):
            raise ValueError('opening registration requires one coherent durable client')
        if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 120:
            raise ValueError('opening review freshness must be within 1..120 seconds')
        self.store, self.snapshots, self.clock = store, snapshots, clock
        self._client, self.max_age_seconds = store._client, max_age_seconds
        self.journals = DatastoreAccountJournal(store, clock=clock)

    def _scope(self, owner_id, scope_id):
        scope = self.store.find_account_scope(scope_id)
        if scope is None or not owner_id or scope.owner_id != owner_id:
            raise PermissionError('Account owner authentication is required.')
        if scope.environment != 'live' or scope.venue not in {'kalshi', 'polymarket'}:
            raise OpeningConflict('A native live account epoch is required.')
        return scope

    def _key(self, scope):
        return self.store._key(scope.id, 'TwinAccountOpeningApproval', str(scope.account_epoch))

    def _registered(self, scope):
        entity = self._client.get(self._key(scope))
        if entity is None:
            return None
        record = json.loads(entity['payload_json'])
        journal = self.journals.load(scope)
        if (journal is None or entity['fingerprint'] != _digest(record)
                or journal.opening.source_ref != entity['fingerprint']
                or record['owner_id'] != scope.owner_id
                or record['proposal']['account_scope_id'] != scope.id
                or record['proposal']['account_epoch'] != scope.account_epoch
                or record['proposal']['scope_ref'] != _digest(scope.to_storage())
                or journal.opening != _attested_opening(scope, record['proposal'], entity['fingerprint'])):
            raise RuntimeError('Opening approval integrity is unconfirmed.')
        return record, _approved(scope, record, entity['fingerprint'])

    def _fresh(self, snapshot):
        now = self.clock()
        if (not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None
                or not 0 <= (now - snapshot.received_at).total_seconds() <= self.max_age_seconds):
            raise OpeningConflict('The native snapshot must be refreshed before review.')
        return now

    def _candidate(self, scope):
        if self.journals.load(scope) is not None:
            raise OpeningConflict('The account epoch already has an immutable opening.')
        projection = self.store.projection(scope.id)
        snapshot = self.snapshots.load(scope.id)
        if (snapshot is None or snapshot.account_epoch != scope.account_epoch
                or projection.account_epoch != scope.account_epoch
                or snapshot.received_at < scope.created_at):
            raise OpeningConflict('A complete snapshot of the current account epoch is required.')
        self._fresh(snapshot)
        # First-opening registration must never absorb an in-flight local order.
        # Refuse any local command, including terminal commands with held capital.
        if (self.store.commands(scope.id, limit=1) or projection.reserved_cash
                or projection.reserved_max_loss or snapshot.reserved_cash
                or set(snapshot.drift_reasons) - {'journal_uninitialized'}):
            raise OpeningConflict('Existing activity requires reconciliation before opening registration.')
        closed = {'executed', 'filled', 'canceled', 'cancelled', 'expired', 'rejected', 'matched'}
        if any(str(row.get('status', '')).lower() not in closed for row in snapshot.orders):
            raise OpeningConflict('Active or unconfirmed venue orders prevent opening registration.')
        if scope.venue == 'polymarket' and any(
            row.get('status') != 'CONFIRMED'
            and not (row.get('status') == 'FAILED' and Decimal(str(row['fee'])) == 0) for row in snapshot.fills
        ):
            raise OpeningConflict('Unconfirmed venue fills prevent opening registration.')
        if any(not row.quantity and row.basis for row in snapshot.holdings):
            raise OpeningConflict('A closed holding cannot carry unexplained basis.')
        proposal = {
            'status': 'review_required', 'account_scope_id': scope.id, 'account_epoch': scope.account_epoch,
            'venue': scope.venue, 'collateral_asset': scope.collateral_asset,
            'scope_ref': _digest(scope.to_storage()),
            'scope': scope.to_storage(), 'snapshot': json.loads(_canonical_json(snapshot)),
            'snapshot_generation': snapshot.generation, 'observed_at': snapshot.received_at.isoformat(),
            'snapshot_ref': sha256(_canonical_json(snapshot).encode()).hexdigest(),
            'projection_revision': projection.revision, 'cash': str(snapshot.total_cash),
            'fees_paid': str(snapshot.fees_paid),
            'positions': [{'instrument_id': row.instrument_id, 'quantity': str(row.quantity), 'basis': str(row.basis)}
                          for row in snapshot.holdings if row.quantity],
            'external_activity_ids': list(snapshot.external_activity_ids),
            'activity_refs': [list(row) for row in external_activity_refs(snapshot)],
        }
        return snapshot, projection, proposal

    @tracer.start_as_current_span('twin.account.opening.review', record_exception=False, set_status_on_exception=False)
    def review(self, owner_id: str, scope_id: str):
        try:
            with _transaction_scope(self._client):
                scope = self._scope(owner_id, scope_id)
                registered = self._registered(scope)
                if registered is not None:
                    result = registered[1]
                else:
                    _, _, proposal = self._candidate(scope)
                    result = {**proposal, 'review_hash': _digest(proposal)}
            operations.add(1, {'operation': 'review', 'outcome': result['status']})
            return result
        # aqg: top-level boundary -- export fixed status only, never private review data
        except Exception:
            _failed('review')
            raise

    @tracer.start_as_current_span('twin.account.opening.confirm', record_exception=False, set_status_on_exception=False)
    def confirm(self, owner_id: str, scope_id: str, request: OpeningConfirmation):
        try:
            # Validate even callers outside HTTP; no generic dictionary approval API.
            request = OpeningConfirmation.model_validate(request)
            with _transaction_scope(self._client):
                scope = self._scope(owner_id, scope_id)
                registered = self._registered(scope)
                if registered is not None:
                    record, result = registered
                    if (record['review_hash'], record['evidence_ref']) != (request.review_hash, request.evidence_ref):
                        raise OpeningConflict('The account epoch already has a different opening approval.')
                else:
                    snapshot, _, proposal = self._candidate(scope)
                    if _digest(proposal) != request.review_hash:
                        raise OpeningConflict('The account changed after review; review it again.')
                    approved_at = self._fresh(snapshot)
                    record = {'owner_id': owner_id, 'review_hash': request.review_hash,
                              'evidence_ref': request.evidence_ref, 'proposal': proposal,
                              'approved_at': approved_at.isoformat()}
                    encoded = _json(record)
                    if len(encoded.encode()) > 900_000:
                        raise OpeningConflict('The opening approval exceeds its private storage limit.')
                    source_ref = _digest(record)
                    self.journals.initialize(_attested_opening(scope, proposal, source_ref))
                    from analyzing_llm_rationale import datastore_backend as datastore
                    entity = datastore.Entity(key=self._key(scope), exclude_from_indexes=('payload_json',))
                    entity.update(payload_json=encoded, fingerprint=source_ref)
                    self._client.put(entity)
                    if self._fresh(snapshot) < approved_at:
                        raise OpeningConflict('The opening registration clock moved backwards.')
                    # Google transaction reads do not see their own pending writes.
                    # The earlier scope/root/snapshot/command reads form the commit
                    # conflict set; use the known staged result, never read it back.
                    result = _approved(scope, record, source_ref)
            operations.add(1, {'operation': 'confirm', 'outcome': 'registered'})
            logger.info('Owner-approved account opening registered')
            return result
        # aqg: top-level boundary -- no private source, money or backend exception text exported
        except Exception:
            _failed('confirm')
            raise


def create_opening_router(service_factory: Callable[[], AccountOpeningService], *, resolve_owner) -> APIRouter:
    router = APIRouter(prefix='/twin/accounts', tags=['Autonomous Twin'])

    def call(request, scope_id, body=None):
        try:
            owner_id = resolve_owner(request)
            if not owner_id:
                raise PermissionError('Account owner authentication is required.')
            service = service_factory()
            return service.review(owner_id, scope_id) if body is None else service.confirm(owner_id, scope_id, body)
        except HTTPException:
            raise
        except PermissionError:
            raise HTTPException(403, 'Account owner authentication is required.') from None
        except OpeningConflict as exc:
            raise HTTPException(409, str(exc)) from None
        # aqg: top-level boundary -- backend/provider failures never disclose private account data
        except Exception:
            raise HTTPException(503, 'Account opening state is unavailable.') from None

    @router.get('/{scope_id}/opening')
    def review_route(request: Request, scope_id: str = Path(min_length=1, max_length=256)):
        return call(request, scope_id)

    @router.post('/{scope_id}/opening')
    def confirm_route(body: OpeningConfirmation, request: Request,
                      scope_id: str = Path(min_length=1, max_length=256)):
        return call(request, scope_id, body)

    return router
