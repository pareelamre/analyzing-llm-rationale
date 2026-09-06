"""Shared account reservation boundary for confirmed manual orders.

The HTTP layer supplies an already-fresh guardrail snapshot.  This module does
no venue I/O and has no FastAPI dependency, so reserving and claiming remain a
short account-scoped state transition that autonomous execution can share.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Any, Mapping

from .models import AccountScope, ProposalAction, TradeIntent, canonical_instrument_id
from .store import ExecutionCommand, TwinStore, TwinStoreError


class ManualReservationConflict(TwinStoreError):
    """A confirmed manual order lost the account-scoped command claim."""


@dataclass(frozen=True)
class ManualCommandClaim:
    command: ExecutionCommand
    fence: int
    worker_id: str


def _digest(*values: str) -> str:
    return sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def reserve_confirmed_manual_order(
    store: TwinStore,
    *,
    user_id: str,
    venue: str,
    payload: Mapping[str, Any],
    normalized: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    authority_ref: str,
    now: datetime | None = None,
) -> ManualCommandClaim:
    """Reserve and fence one already-confirmed manual order before venue I/O.

    `authority_ref` is a server-generated direct-request or saved-run identity.
    It becomes part of the deterministic intent rather than a display label or
    secret connection value.
    """
    now = now or datetime.now(timezone.utc)
    venue = str(venue).lower()
    owner = _digest("owner", user_id)
    connection = _digest("connection", user_id, venue)
    scope_id = f"manual-scope-{_digest('scope', user_id, venue)[:32]}"
    scope = AccountScope(
        id=scope_id,
        owner_id=f"owner-{owner[:32]}",
        venue=venue,
        venue_account_ref=f"account-{connection[:32]}",
        environment="live",
        collateral_asset="USD",
        connection_ref=f"connection-{connection[:32]}",
        account_epoch=1,
        created_at=now,
    )
    policy = guardrails.get("policy") if isinstance(guardrails.get("policy"), Mapping) else {}
    portfolio = guardrails.get("portfolio") if isinstance(guardrails.get("portfolio"), Mapping) else {}
    available = Decimal(str(portfolio.get("available", "0")))
    loss_limit = Decimal(str(policy.get("max_daily_risk_notional", "0")))
    try:
        store.register_account(scope, venue_available_cash=available, loss_limit=loss_limit)
    except TwinStoreError:
        # The normal path finds the existing account.  Epoch changes stay a
        # hard failure; there is no implicit migration of an account identity.
        projection = store.projection(scope_id)
        if projection.account_epoch != scope.account_epoch:
            raise
    store.refresh_account_capacity(scope_id, venue_available_cash=available, loss_limit=loss_limit)

    action = ProposalAction(f"{str(normalized.get('action') or '').upper()}_{str(normalized.get('outcome') or '').upper()}")
    identifier = normalized.get("ticker") if venue == "kalshi" else normalized.get("token_id")
    if not identifier:
        raise TwinStoreError("manual order does not have a stable venue instrument identifier")
    instrument_id = canonical_instrument_id(
        venue=venue,
        environment="live",
        venue_instrument_id=str(identifier),
    )
    client_order_id = str(payload.get("client_order_id") or authority_ref)
    intent = TradeIntent(
        id=f"manual-intent-{_digest(scope_id, client_order_id)[:32]}",
        account_scope_id=scope_id,
        account_epoch=1,
        instrument_id=instrument_id,
        action=action,
        quantity=Decimal(str(normalized["quantity"])),
        limit_price=Decimal(str(normalized["price"])),
        time_in_force=str(normalized.get("time_in_force") or "manual"),
        forecast_id=None,
        exit_reason=f"manual-confirmation-{_digest(authority_ref)[:24]}",
        policy_version="manual-guardrails-v1",
        strategy_version="manual-confirmed-v1",
        market_version=f"manual-quote-{_digest(str(guardrails.get('quote', {})))[:24]}",
        fee_allowance=Decimal("0"),
        slippage_allowance=Decimal("0"),
        expires_at=now + timedelta(minutes=5),
        created_at=now,
    )
    estimated = Decimal(str(normalized.get("estimated_notional", "0")))
    if estimated <= 0:
        estimated = Decimal(str(payload.get("max_cost") or normalized.get("price", "0"))) * Decimal(str(normalized["quantity"]))
    is_buy = action in {ProposalAction.BUY_YES, ProposalAction.BUY_NO}
    store.reserve_intent(intent, cash=estimated if is_buy else Decimal("0"), max_loss=estimated if is_buy else Decimal("0"), now=now)
    command = store.command_for_intent(intent)
    worker_id = f"manual-worker-{_digest(authority_ref)[:24]}"
    claim = store.claim_command(command.id, worker_id=worker_id, now=now)
    if claim is None:
        raise ManualReservationConflict("manual order already has an active execution claim")
    return ManualCommandClaim(command=command, fence=claim.fence, worker_id=worker_id)
