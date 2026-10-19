"""Confirmed manual orders using the existing native prepared-order backend."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from hashlib import sha256
from typing import Any, Callable, Mapping

from opentelemetry import metrics, trace

from .. import market_data, trading
from .execution import ExecutionBlocked, ExecutionContext, SubmissionDisposition
from .manual import ManualCommandClaim, ManualReservationConflict, _digest, owner_scope_ref
from .market import normalize_market
from .market_capture import _verified_trading_cost
from .models import AccountScope, CommandState, TradeIntent
from .polymarket_execution import reserve_polymarket_intent, submit_polymarket_command
from .store import ReservationPreconditions, TwinStore, TwinStoreError

tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter('twin.manual.native_requests', unit='1')


def require_native_manual_enabled() -> None:
    if not trading._env_bool('FORESEA_ENABLE_NATIVE_MANUAL_TRADING', False):
        raise ExecutionBlocked('native manual execution is not enabled; complete the release checks first')


def require_native_manual_preview(normalized: Mapping[str, Any]) -> None:
    require_native_manual_enabled()
    exchange = normalized.get('exchange_order', {})
    if exchange.get('order_type') not in {'FAK', 'FOK'} or exchange.get('post_only', False) is not False:
        raise ExecutionBlocked('native manual execution supports FAK/FOK only until the live resting-order lifecycle is integrated')


def manual_polymarket_client(credentials: Mapping[str, Any]):
    try:
        if not credentials:
            raise ExecutionBlocked('scoped credentials are required')
        return trading._polymarket_client(credentials)
    except Exception:  # aqg: top-level boundary -- SDK constructor text may contain credential material
        raise ExecutionBlocked('native manual client is unavailable') from None


def _available_cash(client: Any) -> Decimal:
    from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

    try:
        response = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        raw = response['balance']
        amount = Decimal(str(raw))
        if isinstance(raw, bool) or not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
            raise ValueError('balance is not integer collateral units')
        return amount/Decimal('1000000')
    except Exception:  # aqg: top-level boundary -- SDK/auth error text must not escape the private client
        raise ExecutionBlocked('current native collateral balance is unavailable') from None


def capture_manual_polymarket(payload: Mapping[str, Any], *, clock: Callable[[], datetime]):
    """Reuse official public market/book readers; never use caller-supplied books."""
    params = {name: str(payload[field]) for name, field in (('id', 'market_id'), ('slug', 'slug'))
              if payload.get(field)}
    if not params:
        raise ExecutionBlocked('manual Polymarket execution requires market_id or slug')
    rows = market_data._get_json(market_data.POLYMARKET_GAMMA_URL, params=params)
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ExecutionBlocked('manual market lookup must return exactly one market')
    market = rows[0]
    tokens = market.get('clobTokenIds')
    tokens = json.loads(tokens) if isinstance(tokens, str) else tokens
    if not isinstance(tokens, list) or len(tokens) != 2 or len(set(map(str, tokens))) != 2:
        raise ExecutionBlocked('manual market must have two distinct outcome tokens')
    books = {str(token): market_data.fetch_polymarket_orderbook(str(token)) for token in tokens}
    return market, books, clock()


def normalized_manual_capture(market: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]], *, now: datetime):
    captured = normalize_market('polymarket', market, received_at=now, sequence=1, orderbooks=books)
    if not captured.eligible:
        raise ExecutionBlocked('manual market capture is incomplete or ineligible')
    cost = _verified_trading_cost('polymarket', market, captured.snapshot)
    if cost is None:
        raise ExecutionBlocked('manual market fee schedule is not verified')
    instrument = replace(captured.instrument, fee_version=cost.schedule_version)
    snapshot = replace(captured.snapshot, fee_version=cost.schedule_version,
                       id=captured.snapshot.id+'-'+sha256(cost.schedule_version.encode()).hexdigest()[:12])
    return instrument, snapshot, cost


def _manual_scope(store: TwinStore, *, user_id: str, maker: str, now: datetime,
                  available: Decimal, loss_limit: Decimal) -> AccountScope:
    owner = owner_scope_ref(user_id)
    connection = _digest('connection', user_id, 'polymarket')
    legacy_id = 'manual-scope-'+_digest('scope', user_id, 'polymarket')[:32]
    legacy = store.find_account_scope(legacy_id)
    if legacy is not None:
        if legacy.owner_id != owner or legacy.venue != 'polymarket' or legacy.environment != 'live':
            raise ExecutionBlocked('legacy account binding requires operator reconciliation')
        projection = store.projection(legacy.id)
        if projection.reserved_cash or projection.reserved_max_loss or store.has_unresolved_work(legacy.id):
            raise ExecutionBlocked('previous manual account requires reconciliation before native migration')
    canonical_id = 'account-'+_digest(owner, 'polymarket', maker.lower(), 'live', 'USD')[:32]
    scope = store.find_account_scope(canonical_id)
    if scope is not None and (scope.owner_id != owner or scope.venue != 'polymarket'
            or scope.environment != 'live' or scope.collateral_asset != 'USD'
            or scope.venue_account_ref.lower() != maker.lower()):
        raise ExecutionBlocked('canonical maker account binding changed')
    # Advisory duplicate detection only. Strong point/ancestor reads above,
    # not a global eventually consistent query, authorize the known cutover.
    for other in store.account_scopes(owner, limit=100):
        if (other.id not in {canonical_id, legacy_id} and other.venue == 'polymarket'
                and other.environment == 'live' and other.venue_account_ref.lower() == maker.lower()):
            raise ExecutionBlocked('noncanonical maker account requires operator reconciliation')
    scope = scope or AccountScope(
        canonical_id,
        owner, 'polymarket', maker, 'live', 'USD', 'connection-'+connection[:32], 1, now,
    )
    try:
        store.register_account(scope, venue_available_cash=available, loss_limit=loss_limit)
    except TwinStoreError:
        existing = store.account_scope(scope.id)
        if replace(existing, created_at=scope.created_at) != scope:
            raise
        scope = existing  # Concurrent creation preserves the first creation time.
    store.refresh_account_capacity(scope.id, venue_available_cash=available, loss_limit=loss_limit)
    return scope


@tracer.start_as_current_span('twin.manual.polymarket.prepare', record_exception=False)
def prepare_confirmed_polymarket_order(
    store: TwinStore, *, user_id: str, payload: Mapping[str, Any], confirmation: str,
    normalized: Mapping[str, Any], guardrails: Mapping[str, Any], credentials: Mapping[str, Any],
    client: Any, market: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]],
    authority_ref: str, now: datetime, clock: Callable[[], datetime],
) -> ManualCommandClaim:
    """Bind one exact human-confirmed request before any possible venue write."""
    operations.add(1, {'operation':'prepare','outcome':'attempt'})
    if confirmation != trading.CONFIRMATION_PHRASE or not user_id or not authority_ref or not credentials:
        raise ExecutionBlocked('native manual execution requires exact confirmation and scoped credentials')
    require_native_manual_preview(normalized)
    trading.require_execution_enabled({'normalized_order': dict(normalized)}, credentials)
    instrument, snapshot, cost = normalized_manual_capture(market, books, now=now)
    action = str(normalized.get('action') or '').upper()+'_'+str(normalized.get('outcome') or '').upper()
    token = instrument.yes_token_id if action.endswith('_YES') else instrument.no_token_id
    if normalized.get('platform') != 'polymarket' or normalized.get('token_id') != token:
        raise ExecutionBlocked('confirmed outcome token does not match authoritative market')
    quantity, price = Decimal(str(normalized['quantity'])), Decimal(str(normalized['price']))
    # The existing captured rate bounds fees over all possible BUY match prices.
    worst_fee = (quantity*cost.fee_per_share_at(min(price, Decimal('.5')))).quantize(
        Decimal('.01'), rounding=ROUND_CEILING)
    cash = quantity*price+worst_fee if action.startswith('BUY_') else Decimal('0')
    if payload.get('max_cost') is not None and action.startswith('BUY_') and cash > Decimal(str(payload['max_cost'])):
        raise ExecutionBlocked('confirmed max_cost does not cover shares and fees')
    policy, portfolio = guardrails['policy'], guardrails['portfolio']
    scope = _manual_scope(store, user_id=user_id, maker=client.builder.funder, now=now,
        available=min(Decimal(str(portfolio['available'])), _available_cash(client)),
        loss_limit=Decimal(str(policy['max_daily_risk_notional'])))
    request_id = str(payload.get('client_order_id') or '')
    if not request_id or len(request_id) > 128:
        raise ExecutionBlocked('manual request identity is invalid')
    exchange = normalized['exchange_order']
    intent = TradeIntent(
        id='manual-intent-'+_digest(scope.id, request_id)[:32], account_scope_id=scope.id,
        account_epoch=scope.account_epoch, instrument_id=instrument.id, action=action,
        quantity=quantity, limit_price=price, time_in_force=str(exchange['order_type']), forecast_id=None,
        exit_reason='manual-confirmation-'+_digest(authority_ref)[:24],
        policy_version='manual-guardrails-v1', strategy_version='manual-confirmed-v1',
        market_version=snapshot.id, fee_allowance=worst_fee, slippage_allowance=Decimal('0'),
        expires_at=now+timedelta(minutes=5), created_at=now,
    )
    context = ExecutionContext(scope, intent.policy_version, intent.strategy_version,
                               intent.market_version, runtime_live_enabled=True)
    command = reserve_polymarket_intent(store, intent=intent, instrument=instrument, snapshot=snapshot,
        book=books[str(token)], client=client, context=context, now=now, clock=clock,
        cash=cash, max_loss=cash, post_only=exchange.get('post_only', False),
        manual_authority_ref=authority_ref,
        preconditions=ReservationPreconditions(store.projection(scope.id).revision, snapshot.id,
                                               snapshot.received_at, snapshot.stale_after_seconds))
    claim = store.claim_command(command.id, worker_id='manual-worker-'+_digest(authority_ref)[:24], now=clock())
    if claim is None:
        raise ManualReservationConflict('manual request already has an execution claim')
    operations.add(1, {'operation':'prepare','outcome':'claimed'})
    return ManualCommandClaim(store.command_for_intent(intent), intent, scope, claim, instrument)


def _submit_confirmed_polymarket_order(
    store: TwinStore, *, command_claim: ManualCommandClaim, payload: Mapping[str, Any],
    user_id: str, credentials: Mapping[str, Any], confirmation: str,
    load_policy: Callable[[], Mapping[str, Any]], clock: Callable[[], datetime],
    load_credentials: Callable[[], Mapping[str, Any] | None],
) -> dict[str, Any]:
    operations.add(1, {'operation':'submit','outcome':'attempt'})
    if (confirmation != trading.CONFIRMATION_PHRASE or not credentials
            or owner_scope_ref(user_id) != command_claim.scope.owner_id
            or command_claim.native_instrument is None):
        raise ExecutionBlocked('native manual submission authority is missing')
    preview = trading.preview_order(payload, credentials)
    normalized, intent = preview['normalized_order'], command_claim.intent
    expected_token = (command_claim.native_instrument.yes_token_id
                      if intent.action.value.endswith('_YES') else command_claim.native_instrument.no_token_id)
    if (normalized.get('platform') != 'polymarket' or normalized.get('token_id') != expected_token
            or normalized['action'].upper()+'_'+normalized['outcome'].upper() != intent.action.value
            or Decimal(str(normalized['quantity'])) != intent.quantity
            or Decimal(str(normalized['price'])) != intent.limit_price
            or normalized['exchange_order']['order_type'] != intent.time_in_force):
        raise ExecutionBlocked('manual request changed after confirmation')
    client = manual_polymarket_client(credentials)

    def load_context():
        require_native_manual_enabled()
        if load_credentials() != credentials:
            raise ExecutionBlocked('manual connection changed during preparation')
        trading.require_execution_enabled(preview, credentials)
        policy = load_policy()
        projection = store.projection(intent.account_scope_id)
        available = _available_cash(client)
        projection = store.refresh_account_capacity(intent.account_scope_id,
            venue_available_cash=available, loss_limit=projection.loss_limit)
        if projection.reserved_cash > available:
            raise ExecutionBlocked('current venue cash no longer covers reserved commitments')
        if (policy.get('paused') is not False
                or projection.reserved_max_loss > Decimal(str(policy['max_daily_risk_notional']))):
            raise ExecutionBlocked('current manual policy blocks execution')
        scope = store.account_scope(intent.account_scope_id)
        return ExecutionContext(scope, intent.policy_version, intent.strategy_version,
                                intent.market_version, runtime_live_enabled=True)

    def load_instrument():
        market, books, received = capture_manual_polymarket(payload, clock=clock)
        instrument, _, _ = normalized_manual_capture(market, books, now=received)
        return instrument

    outcome = submit_polymarket_command(store, command=command_claim.command,
        intent=intent, claim=command_claim.claim, client=client, load_context=load_context,
        load_instrument=load_instrument, clock=clock)
    if outcome.disposition is SubmissionDisposition.REJECTED:
        if outcome.venue_response is None:
            raise ExecutionBlocked('native manual request blocked before transport')
        acknowledgement = dict(outcome.venue_response['acknowledgement'])
        operations.add(1, {'operation':'submit','outcome':'rejected'})
        return {**preview, 'would_execute':True, 'submitted':False, 'user_id':user_id,
            'venue_response':{'acknowledgement':acknowledgement,
                'body':{'orderID':command_claim.command.client_order_id, 'status':'rejected'}}}
    if outcome.disposition is not SubmissionDisposition.ACKNOWLEDGED or outcome.venue_response is None:
        raise trading.TradingExecutionError('native manual outcome requires reconciliation; do not retry')
    acknowledgement = dict(outcome.venue_response['acknowledgement'])
    operations.add(1, {'operation':'submit','outcome':'acknowledged'})
    # Public audit compatibility: acknowledge submission without inventing a
    # fill from a matched/delayed response. Full native status stays in the ack.
    return {**preview, 'would_execute':True, 'submitted':True, 'user_id':user_id,
        'venue_response':{'acknowledgement':acknowledgement,
            'body':{'orderID':command_claim.command.client_order_id, 'status':'submitted'}}}


@tracer.start_as_current_span('twin.manual.polymarket.submit', record_exception=False)
def submit_confirmed_polymarket_order(
    store: TwinStore, *, command_claim: ManualCommandClaim, payload: Mapping[str, Any],
    user_id: str, credentials: Mapping[str, Any], confirmation: str,
    load_policy: Callable[[], Mapping[str, Any]], clock: Callable[[], datetime],
    load_credentials: Callable[[], Mapping[str, Any] | None],
) -> dict[str, Any]:
    try:
        return _submit_confirmed_polymarket_order(store, command_claim=command_claim,
            payload=payload, user_id=user_id, credentials=credentials, confirmation=confirmation,
            load_policy=load_policy, clock=clock, load_credentials=load_credentials)
    except (ExecutionBlocked, trading.TradingError):
        if owner_scope_ref(user_id) == command_claim.scope.owner_id:
            try:
                store.transition_command(command_claim.command.id, target=CommandState.REJECTED,
                    worker_id=command_claim.worker_id, fence=command_claim.fence, require_unstarted=True)
            except TwinStoreError:
                # Started, legacy-unknown, terminal or re-fenced work retains
                # its existing evidence/hold; cleanup cannot undo a sender.
                pass
        operations.add(1, {'operation':'submit','outcome':'blocked'})
        raise
