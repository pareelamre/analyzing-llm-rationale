"""Private prepared CTF orders using the shared authority, store and send fence.

This module installs no live worker or public route. Signed bodies are private
authorization material; API credentials never enter the stored request.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
from typing import Any, Callable, Mapping

from opentelemetry import metrics, trace

from ..trading import _normalise_polymarket_acknowledgement, _require_bounded_acknowledgement
from .execution import (
    ExecutionBlocked,
    ExecutionContext,
    PreDispatchBlocked,
    SubmissionResult,
    _assert_authorized,
    assert_intent_authority,
    submit_claimed_command,
)
from .models import Completeness, Instrument, MarketSnapshot, TradeIntent, canonical_instrument_id
from .recovery import VenueOrderLookup, _require_current_scope, cancel_after_reconciliation
from .store import (
    CommandClaim,
    ExecutionCommand,
    ReservationPreconditions,
    TwinStore,
    TwinStoreError,
    command_request_fingerprint,
    require_durable_store,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter("twin.native_orders", unit="1")
_SDK_VERSIONS = frozenset({"1.1.0", "1.2.0"})
_CTF_CAPABILITIES = frozenset({"ctf-v2", "polymarket-clob-limit"})
_EXCHANGES = {
    False: "0xE111180000d2663C0091e4f400237545B87B996B",
    True: "0xe2222d279d744050d28e00520010520000310F59",
}


def _units(value: Decimal) -> str:
    amount = value * 1_000_000
    if not amount.is_finite() or amount != amount.to_integral_value():
        raise ExecutionBlocked("native order precision is unsupported")
    return str(int(amount))


def _fresh(received: datetime, age: int, now: datetime) -> None:
    if (received.tzinfo is None or now.tzinfo is None or type(age) is not int or age <= 0
            or not 0 <= (now-received).total_seconds() <= age):
        raise ExecutionBlocked("native order market capture is stale")


def _budget_scope(store: TwinStore, intent: TradeIntent, context: ExecutionContext) -> None:
    if not context.autonomous:
        return  # Confirmed manual requests use the account-root server policy.
    if context.mandate is None:
        raise ExecutionBlocked("native order mandate is unavailable")
    ceiling = min(Decimal(context.mandate.max_capital), Decimal(context.mandate.max_loss))
    projection = store.projection(intent.account_scope_id)
    if (projection.loss_limit > ceiling or projection.reserved_max_loss > ceiling
            or projection.reserved_cash > ceiling):
        raise ExecutionBlocked("account reservation limits exceed current mandate")


def _restore(
    payload: Mapping[str, Any], intent: TradeIntent, client: Any, *, now: datetime,
    for_submission: bool = True,
):
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    from py_clob_client_v2.order_utils.exchange_order_builder_v2 import ExchangeOrderBuilderV2
    from py_clob_client_v2.order_utils.model.order_data_v2 import SignedOrderV2

    expected_keys = {"schema_version", "sdk_version", "protocol", "chain_id", "neg_risk",
                     "intent_hash", "market_version", "market_received_at", "stale_after_seconds",
                     "condition_id", "venue_instrument_id", "authority_mode", "post_only", "order_type", "order",
                     "fee_version", "tick_size", "min_quantity", "authority_ref_hash"}
    installed = version("py-clob-client-v2")
    if (set(payload) != expected_keys or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 2 or installed not in _SDK_VERSIONS
            or payload["sdk_version"] not in _SDK_VERSIONS
            or (for_submission and payload["sdk_version"] != installed)
            or payload["protocol"] != "ctf_v2"
            or payload["authority_mode"] not in {"autonomous", "manual"}
            or type(payload["post_only"]) is not bool
            or payload["chain_id"] != 137 or client.signer.get_chain_id() != 137
            or client.host != "https://clob.polymarket.com"
            or type(payload["neg_risk"]) is not bool
            or payload["intent_hash"] != intent.intent_hash
            or payload["market_version"] != intent.market_version
            or not isinstance(payload["condition_id"], str) or not payload["condition_id"]):
        raise ExecutionBlocked("native prepared context does not match")
    if payload["authority_mode"] == "manual" and (
            not isinstance(payload["authority_ref_hash"], str) or len(payload["authority_ref_hash"]) != 64
            or intent.exit_reason != "manual-confirmation-"+payload["authority_ref_hash"][:24]):
        raise ExecutionBlocked("native manual authority binding changed")
    if intent.instrument_id != canonical_instrument_id(
        venue="polymarket", environment="live", venue_instrument_id=payload["venue_instrument_id"],
        condition_id=payload["condition_id"],
    ):
        raise ExecutionBlocked("native condition does not match intent instrument")
    if for_submission:
        _fresh(datetime.fromisoformat(payload["market_received_at"]), payload["stale_after_seconds"], now)
    tif = "FAK" if intent.time_in_force.upper() == "IOC" else intent.time_in_force.upper()
    allowed_tif = {"FAK", "FOK"}
    if not for_submission and payload["authority_mode"] == "manual":
        allowed_tif.add("GTC")  # Read/cancel old packets without authorizing a new resting order.
    if tif not in allowed_tif or payload["order_type"] != tif or (payload["post_only"] and tif != "GTC"):
        raise ExecutionBlocked("native order must be bounded immediate execution")
    wire = dict(payload["order"])
    expected_side = "BUY" if intent.action.value.startswith("BUY_") else "SELL"
    if (wire.get("side") != expected_side or wire.get("expiration") != "0"
            or wire.get("maker", "").lower() != client.builder.funder.lower()
            or wire.get("signer", "").lower() != client.signer.address().lower()
            or wire.get("signatureType") not in (0, 1, 2)
            or wire["signatureType"] != int(client.builder.signature_type)):
        raise ExecutionBlocked("native signer, account or action changed")
    shares, cash = _units(intent.quantity), _units(intent.quantity * intent.limit_price)
    expected = (cash, shares) if expected_side == "BUY" else (shares, cash)
    if (wire.get("makerAmount"), wire.get("takerAmount")) != expected:
        raise ExecutionBlocked("native signed amounts do not match intent")
    signed = SignedOrderV2(**{**wire, "salt": str(wire["salt"]),
                             "side": 0 if expected_side == "BUY" else 1})
    builder = ExchangeOrderBuilderV2(_EXCHANGES[payload["neg_risk"]], 137, client.signer)
    typed = builder.build_order_typed_data(signed)
    if Account.recover_message(encode_typed_data(full_message=typed), signature=signed.signature).lower() != signed.signer.lower():
        raise ExecutionBlocked("native signature does not match signer")
    return signed, builder.build_order_hash(typed)


def _read_request(command: ExecutionCommand, intent: TradeIntent, client: Any, *,
                  now: datetime, for_submission: bool = True):
    encoded = command.prepared_request_json
    if (not encoded or len(encoded.encode("utf-8")) > 64_000
            or command.scope_id != intent.account_scope_id or command.intent_hash != intent.intent_hash
            or command.request_fingerprint != command_request_fingerprint(intent, command.client_order_id, encoded)):
        raise ExecutionBlocked("native request fingerprint changed")
    payload = json.loads(encoded)
    signed, order_hash = _restore(payload, intent, client, now=now, for_submission=for_submission)
    if order_hash != command.client_order_id:
        raise ExecutionBlocked("native order identity changed")
    return payload, signed


@tracer.start_as_current_span("twin.polymarket.prepare", record_exception=False)
def reserve_polymarket_intent(
    store: TwinStore, *, intent: TradeIntent, instrument: Instrument, snapshot: MarketSnapshot,
    book: Mapping[str, Any], client: Any, context: ExecutionContext, now: datetime,
    cash: Decimal, max_loss: Decimal, preconditions: ReservationPreconditions,
    clock: Callable[[], datetime] | None = None,
    post_only: bool = False,
    manual_authority_ref: str | None = None,
) -> ExecutionCommand:
    """Sign once outside transactions, then atomically reserve the exact body."""
    from py_clob_client_v2.client import order_to_json_v2
    from py_clob_client_v2.clob_types import OrderArgs, PartialCreateOrderOptions

    clock = clock or (lambda: datetime.now(timezone.utc))
    span = trace.get_current_span()
    span.set_attributes({"twin.venue": "polymarket", "twin.protocol": "ctf_v2"})
    stage = "authority"
    try:
        require_durable_store(store, live=True)
        assert_intent_authority(intent, context, now=now, cash=cash, max_loss=max_loss)
        if max_loss < cash:
            raise ExecutionBlocked("native loss hold must cover the full cash reservation")
        stage = "instrument"
        if (context.simulation or context.scope.environment != "live"
                or context.scope.venue != "polymarket" or context.scope.collateral_asset != "USD"
                or instrument.venue != "polymarket"
                or instrument.capability_version not in _CTF_CAPABILITIES
                or instrument.environment != "live" or instrument.id != intent.instrument_id
                or snapshot.instrument_id != instrument.id or snapshot.id != intent.market_version
                or snapshot.complete is not Completeness.COMPLETE
                or snapshot.fee_version != instrument.fee_version or instrument.close_at <= now
                or instrument.status not in {"active", "open", "live", "trading"}):
            raise ExecutionBlocked("native order instrument or context is ineligible")
        if not context.autonomous and (intent.strategy_version != "manual-confirmed-v1"
                or intent.policy_version != "manual-guardrails-v1" or not manual_authority_ref
                or intent.exit_reason != "manual-confirmation-"+sha256(manual_authority_ref.encode()).hexdigest()[:24]):
            raise ExecutionBlocked("native manual context requires confirmed manual intent")
        stage = "account"
        if (store.account_scope(context.scope.id) != context.scope
                or context.scope.venue_account_ref.lower() != client.builder.funder.lower()):
            raise ExecutionBlocked("native client does not match registered maker account")
        stage = "budget"
        _budget_scope(store, intent, context)
        stage = "book"
        _fresh(snapshot.received_at, snapshot.stale_after_seconds, now)
        token = instrument.yes_token_id if intent.action.value.endswith("_YES") else instrument.no_token_id
        if (not token or not token.isascii() or not token.isdecimal() or not 0 <= int(token) < 2**256
                or str(int(token)) != token
                or book.get("asset_id") != token or book.get("market") != instrument.condition_id
                or "version" in book or type(book.get("neg_risk")) is not bool
                or Decimal(str(book.get("tick_size"))) != instrument.tick_size
                or Decimal(str(book.get("min_order_size"))) != instrument.min_quantity
                or not 0 < intent.limit_price < 1 or intent.limit_price % instrument.tick_size
                or intent.quantity < instrument.min_quantity or intent.quantity % Decimal(".01")
                or preconditions.market_version != snapshot.id
                or preconditions.market_received_at != snapshot.received_at
                or preconditions.market_stale_after_seconds != snapshot.stale_after_seconds):
            raise ExecutionBlocked("native book, token or precision is unsupported")
        existing = store.find_command_for_intent(intent)
        if existing is not None:
            if (not existing.prepared_request_json or existing.request_fingerprint !=
                    command_request_fingerprint(intent, existing.client_order_id, existing.prepared_request_json)):
                raise ExecutionBlocked("existing native preparation is unavailable or changed")
            operations.add(1, {"operation": "prepare", "outcome": "reused"})
            span.set_attribute("outcome", "reused")
            return existing
        stage = "sdk"
        installed = version("py-clob-client-v2")
        if installed not in _SDK_VERSIONS or client.host != "https://clob.polymarket.com":
            raise ExecutionBlocked("native SDK version has not been verified")
        tif = "FAK" if intent.time_in_force.upper() == "IOC" else intent.time_in_force.upper()
        allowed_tif = {"FAK", "FOK"}
        if tif not in allowed_tif or type(post_only) is not bool or (post_only and tif != "GTC"):
            raise ExecutionBlocked("native order must be immediate")
        signed = client.create_order(
            OrderArgs(token_id=token, price=float(intent.limit_price), size=float(intent.quantity),
                      side="BUY" if intent.action.value.startswith("BUY_") else "SELL"),
            PartialCreateOrderOptions(tick_size=format(instrument.tick_size, "f"), neg_risk=book["neg_risk"]),
        )
        # Do not persist owner/API-key credentials from the HTTP envelope.
        payload = {
            "schema_version": 2, "sdk_version": installed, "protocol": "ctf_v2", "chain_id": 137,
            "neg_risk": book["neg_risk"], "intent_hash": intent.intent_hash,
            "market_version": snapshot.id, "market_received_at": snapshot.received_at.isoformat(),
            "stale_after_seconds": snapshot.stale_after_seconds, "order_type": tif,
            "condition_id": instrument.condition_id,
            "venue_instrument_id": instrument.venue_instrument_id,
            "fee_version": instrument.fee_version, "tick_size": str(instrument.tick_size),
            "min_quantity": str(instrument.min_quantity),
            "authority_mode": "autonomous" if context.autonomous else "manual", "post_only": post_only,
            "authority_ref_hash": sha256(manual_authority_ref.encode()).hexdigest() if manual_authority_ref else None,
            "order": order_to_json_v2(signed, "", tif)["order"],
        }
        completed_at = clock()
        stage = "signed_request"
        assert_intent_authority(intent, context, now=completed_at, cash=cash, max_loss=max_loss)
        _, order_hash = _restore(payload, intent, client, now=completed_at)
        if payload["order"]["tokenId"] != token:
            raise ExecutionBlocked("native signed token changed")
        stage = "persistence"
        store.reserve_intent(intent, cash=cash, max_loss=max_loss, now=completed_at,
                             client_order_id=order_hash, prepared_request=payload,
                             preconditions=preconditions, authority_id=None if context.autonomous else manual_authority_ref,
                             authority_alias=None if context.autonomous else intent.id)
        command = store.command_for_intent(intent)
        operations.add(1, {"operation": "prepare", "outcome": "reserved"})
        span.set_attribute("outcome", "reserved")
        return command
    except Exception as exc:  # aqg: top-level boundary -- never expose signed bodies or credential errors
        reason = stage if isinstance(exc, (ExecutionBlocked, TwinStoreError)) else "unexpected"
        operations.add(1, {"operation": "prepare", "outcome": "blocked"})
        span.set_attributes({"outcome": "blocked", "blocked_reason": reason})
        span.record_exception(ValueError("native preparation blocked"))
        logger.warning("native Polymarket preparation blocked stage=%s", reason)
        raise ExecutionBlocked(f"native Polymarket preparation blocked ({reason})") from None


def submit_polymarket_command(
    store: TwinStore, *, command: ExecutionCommand, intent: TradeIntent, claim: CommandClaim,
    client: Any, load_context: Callable[[], ExecutionContext], load_instrument: Callable[[], Instrument],
    clock: Callable[[], datetime],
) -> SubmissionResult:
    """Restore the private request; shared authority and durable send fence own dispatch."""
    def post(current: ExecutionCommand):
        transport_started = False
        try:
            payload, signed = _read_request(current, intent, client, now=clock())
            instrument = load_instrument()
            token = instrument.yes_token_id if intent.action.value.endswith("_YES") else instrument.no_token_id
            if (instrument.id != intent.instrument_id or instrument.condition_id != payload["condition_id"]
                    or instrument.fee_version != payload["fee_version"]
                    or instrument.tick_size != Decimal(payload["tick_size"])
                    or instrument.min_quantity != Decimal(payload["min_quantity"])
                    or instrument.close_at <= clock() or instrument.status not in {"active", "open", "live", "trading"}
                    or instrument.capability_version not in _CTF_CAPABILITIES or payload["order"]["tokenId"] != token):
                raise ExecutionBlocked("native request no longer matches authoritative instrument")
            # Refresh authority after restoration and immediately before I/O.
            fresh = store.command_for_intent(intent)
            context = load_context()
            if (context.scope.venue != "polymarket" or context.simulation
                    or payload["authority_mode"] != ("autonomous" if context.autonomous else "manual")
                    or context.scope.environment != "live" or store.account_scope(context.scope.id) != context.scope
                    or context.scope.venue_account_ref.lower() != client.builder.funder.lower()):
                raise ExecutionBlocked("native dispatch context is ineligible")
            _assert_authorized(command=fresh, intent=intent, claim=claim,
                               reservation=store.reservation(fresh.scope_id, fresh.reservation_id),
                               context=context, now=clock())
            _budget_scope(store, intent, context)
            _fresh(datetime.fromisoformat(payload["market_received_at"]), payload["stale_after_seconds"], clock())
            transport_started = True
            response = _require_bounded_acknowledgement(
                client.post_order(signed, payload["order_type"], post_only=payload["post_only"], defer_exec=False),
                venue="Polymarket",
            )
            if "success" in response and "ok" in response and response["success"] != response["ok"]:
                raise ExecutionBlocked("native acknowledgement flags conflict")
            if response.get("success", response.get("ok")) is False:
                identities = (response.get(k) for k in ("orderID", "order_id", "id"))
                trade_ids = (response.get(k) for k in ("tradeIDs", "trade_ids"))
                quantities = [response.get(k) for k in (
                    "makingAmount", "takingAmount", "making_amount", "taking_amount",
                )]
                if (any(value not in (None, "") for value in identities)
                        or any(value is not None and value != [] for value in trade_ids)
                        or any(value not in (None, "") and (
                            isinstance(value, bool) or Decimal(str(value)) != 0
                        ) for value in quantities)
                        or str(response.get("status") or "").lower() not in {"", "rejected", "invalid", "denied"}
                        or not isinstance(response.get("errorMsg", response.get("message")), str)
                        or not response.get("errorMsg", response.get("message"))):
                    raise ExecutionBlocked("native rejection contains contradictory acceptance evidence")
                operations.add(1, {"operation": "submit", "outcome": "rejected"})
                return {"acknowledgement": {"confirmed_rejection": True, "status": "rejected"}}
            ack = _normalise_polymarket_acknowledgement(response)
            if ack["venue_order_id"] != current.client_order_id:
                raise ExecutionBlocked("native acknowledgement identity mismatch")
            operations.add(1, {"operation": "submit", "outcome": "acknowledged"})
            return {"acknowledgement": ack}
        except Exception:  # aqg: top-level boundary -- preserve ambiguity without secret-bearing SDK text
            if not transport_started:
                operations.add(1, {"operation": "submit", "outcome": "blocked_before_transport"})
                raise PreDispatchBlocked("native Polymarket transport was not called") from None
            operations.add(1, {"operation": "submit", "outcome": "unknown"})
            logger.warning("native Polymarket dispatch requires reconciliation")
            raise ExecutionBlocked("native Polymarket dispatch outcome is uncertain") from None

    return submit_claimed_command(store, command=command, intent=intent, claim=claim,
                                   context=load_context(), now=clock(), submit=post)


@tracer.start_as_current_span("twin.polymarket.lookup", record_exception=False)
def lookup_polymarket_command(
    store: TwinStore, *, command: ExecutionCommand, intent: TradeIntent,
    client: Any, clock: Callable[[], datetime],
) -> VenueOrderLookup:
    """Prove positive presence only; one missing/error response cannot release cash."""
    complete = False
    try:
        scope = _require_current_scope(store, command, intent)
        if scope.venue_account_ref.lower() != client.builder.funder.lower():
            raise ExecutionBlocked("native recovery account changed")
        current = store.command_for_intent(intent)
        if current.id != command.id:
            raise ExecutionBlocked("recovery command identity changed")
        payload, _ = _read_request(current, intent, client, now=clock(), for_submission=False)
        body = client.get_order(current.client_order_id)
        wire = payload["order"]
        if (not isinstance(body, Mapping) or body.get("id") != current.client_order_id
                or body.get("asset_id") != wire["tokenId"] or body.get("market") != payload["condition_id"]
                or str(body.get("maker_address") or "").lower() != wire["maker"].lower()
                or body.get("side") != wire["side"] or Decimal(str(body.get("price"))) != intent.limit_price
                or Decimal(str(body.get("original_size"))) != intent.quantity
                or not Decimal("0") <= Decimal(str(body.get("size_matched"))) <= intent.quantity):
            raise ExecutionBlocked("native lookup is incomplete or mismatched")
        complete = True
    except Exception:  # aqg: top-level boundary -- errors and 404s remain unknown, never proof of absence
        logger.warning("native Polymarket lookup incomplete")
    operations.add(1, {"operation": "lookup", "outcome": "found" if complete else "unknown"})
    trace.get_current_span().set_attributes({"twin.venue": "polymarket",
                                           "outcome": "found" if complete else "unknown"})
    return VenueOrderLookup(command.scope_id, intent.instrument_id, command.client_order_id,
                            command.request_fingerprint, complete, True if complete else None, clock())


def cancel_polymarket_command(
    store: TwinStore, *, command: ExecutionCommand, intent: TradeIntent, claim: CommandClaim,
    lookup: VenueOrderLookup, client: Any, clock: Callable[[], datetime],
) -> Mapping[str, Any]:
    """Use the existing fresh reconciliation/fence gate; cancellation never releases cash."""
    def cancel(current):
        from py_clob_client_v2.clob_types import OrderPayload

        try:
            scope = _require_current_scope(store, current, intent)
            _read_request(current, intent, client, now=clock(), for_submission=False)
            latest = store.command_for_intent(intent)
            checked_at = clock()
            if (scope.venue_account_ref.lower() != client.builder.funder.lower()
                    or latest.claim != claim or claim.lease_expires_at <= checked_at
                    or not 0 <= (checked_at-lookup.observed_at).total_seconds() <= 5):
                raise ExecutionBlocked("native cancellation account, fence or lookup changed")
            response = client.cancel_order(OrderPayload(orderID=current.client_order_id))
            if (not isinstance(response, Mapping) or response.get("canceled") != [current.client_order_id]
                    or not isinstance(response.get("not_canceled"), Mapping)
                    or current.client_order_id in response["not_canceled"]):
                raise ExecutionBlocked("native cancellation is unconfirmed")
            # This is a cancel acknowledgement. Lifecycle reconciliation still
            # owns fills, final cancellation and reservation release.
            operations.add(1, {"operation": "cancel", "outcome": "acknowledged"})
            return {"canceled": True, "venue_order_id": current.client_order_id}
        except Exception:  # aqg: top-level boundary -- never include a signed body or SDK error text
            logger.warning("native Polymarket cancellation requires reconciliation")
            raise ExecutionBlocked("native Polymarket cancellation outcome is uncertain") from None

    return cancel_after_reconciliation(store, command=command, intent=intent, claim=claim,
                                       lookup=lookup, cancel=cancel, now=clock())
