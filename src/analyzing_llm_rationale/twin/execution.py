"""Authority-checked, fenced submission for the autonomous trading twin.

The caller creates a reservation and command before this module runs.  This
module never chooses a client order identity, retries a venue call, or promotes
an acknowledgement to a fill.  A loss of response is durable
``submission_unknown`` state for reconciliation, not permission to submit a
second order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from .mandates import Mandate
from .models import AccountScope, CommandState, TradeIntent
from .store import CommandClaim, ExecutionCommand, TwinStore, TwinStoreError, require_durable_store


class ExecutionBlocked(RuntimeError):
    """An authority, freshness, or state precondition prevented a venue write."""


class SubmissionUnknown(RuntimeError):
    """The venue may have accepted a request whose outcome was not persisted."""


class SubmissionDisposition(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"
    UNKNOWN = "submission_unknown"
    ALREADY_PROCESSED = "already_processed"


@dataclass(frozen=True)
class ExecutionContext:
    """Server-derived facts rechecked immediately before dispatch.

    ``autonomous`` cannot be asserted by an HTTP caller: routes and workers
    construct this record after authenticating their own manual approval or
    mandate lookup.  Manual commands intentionally do not need forecast
    readiness; autonomous new exposure does.
    """

    scope: AccountScope
    policy_version: str
    strategy_version: str
    market_version: str
    runtime_live_enabled: bool
    autonomous: bool = False
    mandate: Optional[Mandate] = None
    readiness_hash: Optional[str] = None


@dataclass(frozen=True)
class SubmissionResult:
    command: ExecutionCommand
    disposition: SubmissionDisposition
    venue_response: Optional[Mapping[str, Any]] = None


VenueSubmitter = Callable[[ExecutionCommand], Mapping[str, Any]]


def _assert_authorized(
    *,
    command: ExecutionCommand,
    intent: TradeIntent,
    claim: CommandClaim,
    context: ExecutionContext,
    now: datetime,
) -> None:
    if now.tzinfo is None:
        raise ExecutionBlocked("execution time must be timezone-aware")
    if command.id != claim.command_id or command.scope_id != context.scope.id:
        raise ExecutionBlocked("command claim does not match the active account scope")
    if command.intent_id != intent.id or command.intent_hash != intent.intent_hash:
        raise ExecutionBlocked("command is not bound to the supplied immutable intent")
    if not command.request_fingerprint:
        raise ExecutionBlocked("command has no persisted request fingerprint")
    if command.claim is None or command.claim.worker_id != claim.worker_id or command.claim.fence != claim.fence:
        raise ExecutionBlocked("execution claim is no longer current")
    if command.claim.lease_expires_at <= now:
        raise ExecutionBlocked("execution claim has expired")
    if intent.expires_at <= now:
        raise ExecutionBlocked("trade intent has expired")
    try:
        intent.assert_authorization_context(
            context.scope,
            policy_version=context.policy_version,
            strategy_version=context.strategy_version,
            market_version=context.market_version,
        )
    except Exception as exc:
        raise ExecutionBlocked(str(exc)) from exc

    mandate = context.mandate
    if context.autonomous:
        if mandate is None or not mandate.active(now=now):
            raise ExecutionBlocked("autonomous mandate is inactive")
        if mandate.account_scope_id != context.scope.id or mandate.account_epoch != context.scope.account_epoch:
            raise ExecutionBlocked("mandate account binding is stale")
        if mandate.strategy_version != intent.strategy_version:
            raise ExecutionBlocked("mandate strategy version is stale")
        if intent.action.value not in mandate.allowed_actions:
            raise ExecutionBlocked("mandate does not allow this trade action")
        if not context.readiness_hash or context.readiness_hash != mandate.readiness_hash:
            raise ExecutionBlocked("autonomous strategy readiness is missing or stale")
    elif mandate is not None:
        # A manual confirmation may carry an audit mandate reference, but it
        # must never silently become autonomous authority.
        raise ExecutionBlocked("manual command cannot carry autonomous mandate authority")

    is_live = context.scope.environment == "live"
    if is_live and not context.runtime_live_enabled:
        raise ExecutionBlocked("live execution remains disabled")
    if context.autonomous and mandate is not None and mandate.live != is_live:
        raise ExecutionBlocked("mandate environment does not match account scope")


def _classify_venue_response(response: Mapping[str, Any]) -> SubmissionDisposition:
    """Accept only a normalized acknowledgement; never infer a fill here."""
    if not isinstance(response, Mapping):
        return SubmissionDisposition.UNKNOWN
    acknowledgement = response.get("acknowledgement", response)
    if not isinstance(acknowledgement, Mapping):
        return SubmissionDisposition.UNKNOWN
    status = str(acknowledgement.get("status") or acknowledgement.get("venue_status") or "").lower()
    if acknowledgement.get("confirmed_rejection") is True or status in {"rejected", "denied", "invalid"}:
        return SubmissionDisposition.REJECTED
    if acknowledgement.get("acknowledged") is True or status in {"acknowledged", "accepted", "matched", "open", "live", "delayed"}:
        return SubmissionDisposition.ACKNOWLEDGED
    return SubmissionDisposition.UNKNOWN


def submit_claimed_command(
    store: TwinStore,
    *,
    command: ExecutionCommand,
    intent: TradeIntent,
    claim: CommandClaim,
    context: ExecutionContext,
    now: datetime,
    submit: VenueSubmitter,
) -> SubmissionResult:
    """Send one pre-reserved, fenced command at most once.

    The exact client order ID and request fingerprint are stored by
    ``reserve_intent``.  Calls arriving after an acknowledgement, rejection,
    or ambiguity return the durable state without touching the venue.
    """
    if context.scope.environment == "live":
        require_durable_store(store, live=True)
    if command.intent_id != intent.id or command.intent_hash != intent.intent_hash:
        raise ExecutionBlocked("command is not bound to the supplied immutable intent")
    current = store.command_for_intent(intent)
    if current.id != command.id:
        raise ExecutionBlocked("command identity does not match the reserved intent")
    if current.state is not CommandState.SUBMITTING:
        if current.state in {
            CommandState.ACKNOWLEDGED,
            CommandState.REJECTED,
            CommandState.SUBMISSION_UNKNOWN,
            CommandState.PARTIALLY_FILLED,
            CommandState.FILLED,
            CommandState.CANCEL_REQUESTED,
            CommandState.CANCELLED,
        }:
            return SubmissionResult(current, SubmissionDisposition.ALREADY_PROCESSED)
        raise ExecutionBlocked(f"command is not dispatchable from state {current.state.value}")

    _assert_authorized(command=current, intent=intent, claim=claim, context=context, now=now)
    try:
        response = submit(current)
    except Exception as exc:
        try:
            updated = store.transition_command(
                current.id, target=CommandState.SUBMISSION_UNKNOWN, fence=claim.fence, worker_id=claim.worker_id
            )
        except TwinStoreError as transition_error:
            raise SubmissionUnknown("venue response was lost and command state could not be recorded") from transition_error
        raise SubmissionUnknown("venue response was lost; reconcile the persisted order identity") from exc

    disposition = _classify_venue_response(response)
    target = {
        SubmissionDisposition.ACKNOWLEDGED: CommandState.ACKNOWLEDGED,
        SubmissionDisposition.REJECTED: CommandState.REJECTED,
        SubmissionDisposition.UNKNOWN: CommandState.SUBMISSION_UNKNOWN,
    }[disposition]
    try:
        updated = store.transition_command(current.id, target=target, fence=claim.fence, worker_id=claim.worker_id)
    except TwinStoreError as exc:
        # The venue call happened.  Do not retry it if persistence races or
        # fails; recovery must search using the stored identity/fingerprint.
        raise SubmissionUnknown("venue response received but command persistence failed") from exc
    return SubmissionResult(updated, disposition, dict(response))


def submit_authorized_command(
    mandate: Mandate,
    *,
    now: datetime,
    live_enabled: bool,
    submit: Callable[[], object],
) -> object:
    """Compatibility wrapper for the original shadow-only façade.

    New runtime code must use :func:`submit_claimed_command`, whose state and
    authority bindings cannot be supplied by a public request boolean.
    """
    if not mandate.active(now=now):
        raise ExecutionBlocked("mandate is inactive")
    if mandate.live and not live_enabled:
        raise ExecutionBlocked("live execution remains disabled")
    return submit()
