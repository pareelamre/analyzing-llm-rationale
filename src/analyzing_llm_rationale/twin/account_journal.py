"""Private account replay facts, never spending authority or balancing adjustments.

Only trusted source adapters/operator procedures may initialize/import facts.
This module does not attest that a source digest is a verified venue receipt.
No route, tool or task accepts raw journal facts. The separate owner review flow
may attest a first baseline; there is no automatic snapshot bootstrap.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import Callable

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .account import AccountSnapshot, _account_sum, external_activity_refs
from .models import AccountScope
from .store import DatastoreTwinStore, _transaction_scope

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter('twin.account.journal', unit='1')


class AccountJournalError(RuntimeError):
    """Unconfirmed account history; preserve holds and require reconciliation."""


@dataclass(frozen=True)
class JournalPosition:
    instrument_id: str
    quantity: Decimal
    basis: Decimal


@dataclass(frozen=True)
class JournalOpening:
    scope: AccountScope
    observed_at: datetime
    cash: Decimal
    positions: tuple[JournalPosition, ...]
    fees_paid: Decimal
    source_ref: str
    adopted_activity: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class JournalFact:
    id: str
    version: int
    kind: str
    occurred_at: datetime
    cash_delta: Decimal
    positions: tuple[JournalPosition, ...]
    fee: Decimal
    source_ref: str
    external_cash_flow: Decimal = Decimal('0')
    final: bool = True
    sequence: int = 0
    voided: bool = False


@dataclass(frozen=True)
class AccountJournal:
    opening: JournalOpening
    facts: tuple[JournalFact, ...] = ()
    revision: int = 1

    @property
    def fingerprint(self) -> str:
        return sha256(_encode(self).encode()).hexdigest()


@dataclass(frozen=True)
class JournalProjection:
    cash: Decimal
    positions: tuple[JournalPosition, ...]
    fees_paid: Decimal
    external_cash_flow: Decimal
    realized_pnl: Decimal
    pending_settlement: bool


def _decimal(value, *, nonnegative=False):
    if (not isinstance(value, Decimal) or not value.is_finite() or (nonnegative and value < 0)
            or len(value.as_tuple().digits) > 36 or not -18 <= value.as_tuple().exponent <= 18):
        raise AccountJournalError('journal_decimal_invalid')
    return value


def _time(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AccountJournalError('journal_timestamp_invalid')


def _positions(rows, *, opening=False):
    if not isinstance(rows, tuple) or len(rows) > 1000:
        raise AccountJournalError('journal_positions_invalid')
    seen = set()
    for row in rows:
        if (not isinstance(row, JournalPosition) or not isinstance(row.instrument_id, str)
                or not row.instrument_id.strip() or row.instrument_id != row.instrument_id.strip()
                or len(row.instrument_id) > 256 or row.instrument_id in seen):
            raise AccountJournalError('journal_position_identity_invalid')
        seen.add(row.instrument_id)
        _decimal(row.quantity)
        _decimal(row.basis, nonnegative=opening)
        if row.quantity == 0:
            raise AccountJournalError('journal_zero_quantity_invalid')


def _source(ref):
    if not isinstance(ref, str) or re.fullmatch('[0-9a-f]{64}', ref) is None:
        raise AccountJournalError('journal_source_reference_invalid')


def _opening(value):
    if not isinstance(value, JournalOpening) or not isinstance(value.scope, AccountScope):
        raise AccountJournalError('journal_opening_invalid')
    if type(value.scope.account_epoch) is not int:
        raise AccountJournalError('journal_scope_invalid')
    _time(value.observed_at)
    if value.observed_at < value.scope.created_at:
        raise AccountJournalError('journal_opening_predates_scope')
    _decimal(value.cash, nonnegative=True)
    _decimal(value.fees_paid, nonnegative=True)
    _positions(value.positions, opening=True)
    _source(value.source_ref)
    if not isinstance(value.adopted_activity, tuple) or len(value.adopted_activity) > 4096:
        raise AccountJournalError('journal_opening_activity_invalid')
    identities = set()
    for item in value.adopted_activity:
        if (not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                or not item[0].strip() or len(item[0]) > 256 or item[0] in identities):
            raise AccountJournalError('journal_opening_activity_invalid')
        identities.add(item[0])
        _source(item[1])


def _fact(value, opening):
    if (not isinstance(value, JournalFact) or not isinstance(value.id, str)
            or not value.id.strip() or value.id != value.id.strip() or len(value.id) > 256
            or type(value.version) is not int or value.version < 1
            or type(value.final) is not bool
            or type(value.voided) is not bool or type(value.sequence) is not int or value.sequence < 0
            or value.kind not in ('fill', 'settlement', 'external', 'fee', 'income')):
        raise AccountJournalError('journal_fact_identity_invalid')
    _time(value.occurred_at)
    if value.occurred_at <= opening.observed_at:
        raise AccountJournalError('journal_fact_predates_opening')
    _source(value.source_ref)
    _positions(value.positions)
    _decimal(value.cash_delta)
    _decimal(value.external_cash_flow)
    _decimal(value.fee, nonnegative=True)
    if value.voided:
        if (value.version < 2 or not value.final or value.cash_delta or value.positions
                or value.fee or value.external_cash_flow):
            raise AccountJournalError('journal_void_invalid')
        return
    if not value.final and value.kind != 'settlement':
        raise AccountJournalError('journal_provisional_kind_invalid')
    if value.kind == 'external':
        if value.positions or value.fee or value.cash_delta != value.external_cash_flow:
            raise AccountJournalError('journal_external_flow_invalid')
    elif value.external_cash_flow:
        raise AccountJournalError('journal_unexplained_external_flow')
    if value.kind == 'fee' and (value.positions or value.cash_delta != -value.fee):
        raise AccountJournalError('journal_fee_invalid')
    if value.kind == 'income' and (value.positions or value.cash_delta < 0 or value.fee):
        raise AccountJournalError('journal_income_invalid')
    if value.kind in ('fill', 'settlement') and not value.positions:
        raise AccountJournalError('journal_inventory_fact_missing')
    if value.kind == 'settlement' and value.cash_delta + value.fee < 0:
        raise AccountJournalError('journal_settlement_cash_invalid')
    if value.kind == 'settlement' and value.cash_delta + value.fee > sum(
        (abs(row.quantity) for row in value.positions), Decimal('0')
    ):
        raise AccountJournalError('journal_settlement_exceeds_binary_payout')


def _append(current, facts):
    _opening(current.opening)
    history = list(current.facts)
    by_identity = {(row.id, row.version): row for row in history}
    latest = {}
    for row in history:
        latest[row.id] = row
    for fact in facts:
        _fact(fact, current.opening)
        prior = by_identity.get((fact.id, fact.version))
        if prior is not None:
            if prior != fact:
                raise AccountJournalError('journal_same_version_conflict')
            continue
        previous = latest.get(fact.id)
        if (fact.version != (1 if previous is None else previous.version+1)
                or previous is not None and (fact.kind, fact.occurred_at, fact.sequence) !=
                (previous.kind, previous.occurred_at, previous.sequence)):
            raise AccountJournalError('journal_version_or_identity_changed')
        history.append(fact)
        latest[fact.id] = fact
        by_identity[fact.id, fact.version] = fact
    if len(history) > 4096:
        raise AccountJournalError('journal_history_limit')
    order = [(row.occurred_at, row.sequence) for row in latest.values()]
    if len(order) != len(set(order)):
        raise AccountJournalError('journal_source_order_ambiguous')
    return replace(current, facts=tuple(history), revision=current.revision+(tuple(history) != current.facts))


def replay_journal(journal: AccountJournal, *, as_of: datetime) -> JournalProjection:
    """Replay newest facts; corrections replace economics, never add another payout."""
    with localcontext() as context:
        context.prec = 80
        return _replay_journal(journal, as_of=as_of)


def _replay_journal(journal: AccountJournal, *, as_of: datetime) -> JournalProjection:
    _opening(journal.opening)
    _time(as_of)
    if as_of < journal.opening.observed_at:
        raise AccountJournalError('journal_opening_not_yet_effective')
    # Validate historical identities and versions even when they were superseded.
    validated = _append(AccountJournal(journal.opening), journal.facts)
    latest = {row.id: row for row in validated.facts}
    positions = {row.instrument_id: row for row in journal.opening.positions}
    cash, fees, flows = journal.opening.cash, journal.opening.fees_paid, Decimal('0')
    pending = False
    for fact in sorted(latest.values(), key=lambda row: (row.occurred_at, row.sequence)):
        if fact.occurred_at > as_of:
            # Future facts mean the caller's read is older than its journal.
            raise AccountJournalError('journal_fact_after_capture')
        if fact.voided:
            continue
        if not fact.final:
            pending = True
            continue
        cash += fact.cash_delta
        fees += fact.fee
        flows += fact.external_cash_flow
        if cash < 0:
            raise AccountJournalError('journal_cash_negative')
        for delta in fact.positions:
            prior = positions.get(delta.instrument_id, JournalPosition(delta.instrument_id, Decimal('0'), Decimal('0')))
            quantity, basis = prior.quantity+delta.quantity, prior.basis+delta.basis
            if basis < 0 or (quantity == 0 and basis != 0):
                raise AccountJournalError('journal_inventory_basis_invalid')
            if fact.kind == 'settlement' and (
                prior.quantity == 0 or prior.quantity*delta.quantity >= 0
                or abs(delta.quantity) > abs(prior.quantity) or delta.basis > 0
            ):
                raise AccountJournalError('journal_settlement_inventory_invalid')
            if quantity == 0:
                positions.pop(delta.instrument_id, None)
            else:
                positions[delta.instrument_id] = JournalPosition(delta.instrument_id, quantity, basis)
    basis = sum((row.basis for row in positions.values()), Decimal('0'))
    opening_basis = sum((row.basis for row in journal.opening.positions), Decimal('0'))
    return JournalProjection(cash, tuple(positions[key] for key in sorted(positions)), fees, flows,
                             cash+basis-journal.opening.cash-opening_basis-flows, pending)


def _encode(journal):
    def encode(value):
        if isinstance(value, Decimal):
            text = format(value, 'f')
            return text.rstrip('0').rstrip('.') if '.' in text else text
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError('unsupported journal value')
    payload = asdict(journal)
    # Empty adoption must retain the fingerprints of already persisted journals.
    if not journal.opening.adopted_activity:
        payload['opening'].pop('adopted_activity')
    return json.dumps(payload, default=encode, sort_keys=True, separators=(',', ':'))


def _restore(raw):
    def positions(rows):
        return tuple(JournalPosition(row['instrument_id'], Decimal(row['quantity']), Decimal(row['basis'])) for row in rows)
    value = json.loads(raw)
    opening = value['opening']
    scope = dict(opening['scope'])
    scope['created_at'] = datetime.fromisoformat(scope['created_at'])
    opened = JournalOpening(AccountScope(**scope), datetime.fromisoformat(opening['observed_at']),
        Decimal(opening['cash']), positions(opening['positions']), Decimal(opening['fees_paid']), opening['source_ref'],
        tuple(tuple(row) for row in opening.get('adopted_activity', ())))
    facts = tuple(JournalFact(row['id'], row['version'], row['kind'], datetime.fromisoformat(row['occurred_at']),
        Decimal(row['cash_delta']), positions(row['positions']), Decimal(row['fee']), row['source_ref'],
        Decimal(row['external_cash_flow']), row['final'], row['sequence'], row['voided']) for row in value['facts'])
    return AccountJournal(opened, facts, value['revision'])


def compare_journal_snapshot(snapshot: AccountSnapshot, journal: AccountJournal | None) -> AccountSnapshot:
    """Compare independent expectations; never seed them from the observed totals."""
    reasons = list(snapshot.drift_reasons)
    if journal is None:
        reasons.append('journal_uninitialized')
        return replace(snapshot, divergence=True, drift_reasons=tuple(sorted(set(reasons))))
    if (journal.opening.scope.id != snapshot.scope_id
            or journal.opening.scope.account_epoch != snapshot.account_epoch):
        raise AccountJournalError('journal_snapshot_scope_changed')
    try:
        expected = replay_journal(journal, as_of=snapshot.received_at)
    except AccountJournalError:
        return replace(snapshot, divergence=True, journal_revision=journal.revision,
            journal_fingerprint=journal.fingerprint,
            drift_reasons=tuple(sorted(set((*reasons, 'journal_time_or_history_unconfirmed')))))
    if expected.pending_settlement:
        reasons.append('journal_settlement_pending')
    # Exact comparisons until a source adapter supplies documented venue precision.
    for name, observed, wanted in (
        ('cash', snapshot.total_cash, expected.cash),
        ('fees', snapshot.fees_paid, expected.fees_paid),
        ('basis', snapshot.position_basis, _account_sum(row.basis for row in expected.positions)),
    ):
        if observed != wanted:
            reasons.append('journal_'+name+'_mismatch')
    observed = {row.instrument_id: (row.quantity, row.basis) for row in snapshot.holdings if row.quantity}
    wanted = {row.instrument_id: (row.quantity, row.basis) for row in expected.positions}
    if {key: value[0] for key, value in observed.items()} != {key: value[0] for key, value in wanted.items()}:
        reasons.append('journal_holdings_mismatch')
    if {key: value[1] for key, value in observed.items()} != {key: value[1] for key, value in wanted.items()}:
        reasons.append('journal_basis_mismatch')
    adopted = dict(journal.opening.adopted_activity)
    unexplained = tuple(identity for identity, ref in external_activity_refs(snapshot) if adopted.get(identity) != ref)
    return replace(snapshot, journal_revision=journal.revision, journal_fingerprint=journal.fingerprint,
                   external_activity_ids=unexplained, divergence=bool(unexplained or reasons),
                   drift_reasons=tuple(sorted(set(reasons))))


class DatastoreAccountJournal:
    """Private, epoch-bound replay journal; all writers need a verified source adapter."""

    def __init__(self, store: DatastoreTwinStore, *, clock: Callable[[], datetime] | None = None):
        self.store, self._client = store, store._client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _key(self, scope):
        return self.store._key(scope.id, 'TwinAccountJournal', str(scope.account_epoch))

    def load(self, scope: AccountScope) -> AccountJournal | None:
        if self.store.account_scope(scope.id) != scope:
            raise AccountJournalError('journal_scope_changed')
        entity = self._client.get(self._key(scope))
        if entity is None:
            return None
        try:
            result = _restore(entity['payload_json'])
            if (result.opening.scope != scope or type(result.revision) is not int or result.revision < 1
                    or result.fingerprint != entity['fingerprint'] or result.revision != entity['revision']):
                raise AccountJournalError('journal_integrity_invalid')
            replay_journal(result, as_of=max((row.occurred_at for row in result.facts), default=result.opening.observed_at))
            return result
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise AccountJournalError('journal_integrity_invalid') from exc

    def _write(self, journal):
        from analyzing_llm_rationale import datastore_backend as datastore
        encoded = _encode(journal)
        if len(encoded.encode()) > 900_000:
            raise AccountJournalError('journal_storage_limit')
        entity = datastore.Entity(key=self._key(journal.opening.scope), exclude_from_indexes=('payload_json',))
        entity.update(payload_json=encoded, fingerprint=journal.fingerprint, revision=journal.revision)
        self._client.put(entity)
        # Invalidate risk decisions already built from a previous journal state.
        # Money, loss limits and all reservation totals remain exactly unchanged.
        root = self._client.get(self.store._key(journal.opening.scope.id))
        projection = self.store._projection(root)
        self.store._write_projection(root, replace(projection, revision=projection.revision+1))
        self._client.put(root)

    @tracer.start_as_current_span('twin.account.journal.initialize', record_exception=False, set_status_on_exception=False)
    def initialize(self, opening: JournalOpening, *, validate: Callable[[], None] | None = None) -> AccountJournal:
        try:
            _opening(opening)
            now = self.clock()
            _time(now)
            if opening.observed_at > now:
                raise AccountJournalError('journal_opening_after_import')
            with _transaction_scope(self._client):
                if validate is not None:
                    validate()
                prior = self.load(opening.scope)
                if prior is not None:
                    if prior.opening != opening:
                        raise AccountJournalError('journal_opening_immutable')
                    return prior
                journal = AccountJournal(opening)
                self._write(journal)
                if validate is not None:
                    validate()
            operations.add(1, {'outcome': 'initialized'})
            logger.info('Private account journal initialized')
            return journal
        # aqg: top-level boundary -- mark fixed errors without exporting private source data
        except Exception:
            operations.add(1, {'outcome': 'unconfirmed'})
            trace.get_current_span().set_status(Status(StatusCode.ERROR, 'journal_unconfirmed'))
            logger.warning('Private account journal initialization requires reconciliation')
            raise

    @tracer.start_as_current_span('twin.account.journal.append', record_exception=False, set_status_on_exception=False)
    def append(self, scope: AccountScope, facts: tuple[JournalFact, ...], *, expected_revision: int,
               validate: Callable[[], None] | None = None) -> AccountJournal:
        try:
            with _transaction_scope(self._client):
                if validate is not None:
                    validate()
                prior = self.load(scope)
                if prior is None or type(expected_revision) is not int or expected_revision != prior.revision:
                    raise AccountJournalError('journal_revision_changed')
                updated = _append(prior, facts)
                now = self.clock()
                _time(now)
                if any(row.occurred_at > now for row in updated.facts):
                    raise AccountJournalError('journal_fact_after_import')
                replay_journal(updated, as_of=max((row.occurred_at for row in updated.facts), default=updated.opening.observed_at))
                if updated != prior:
                    self._write(updated)
                if validate is not None:
                    validate()
            outcome = 'duplicate' if updated == prior else 'stored'
            operations.add(1, {'outcome': outcome})
            trace.get_current_span().set_attribute('outcome', outcome)
            logger.info('Private account journal import outcome=%s', outcome)
            return updated
        # aqg: top-level boundary -- storage/validation errors must not enter span exception text
        except Exception:
            operations.add(1, {'outcome': 'unconfirmed'})
            trace.get_current_span().set_status(Status(StatusCode.ERROR, 'journal_unconfirmed'))
            logger.warning('Private account journal import requires reconciliation')
            raise
