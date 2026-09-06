"""Conservative recovery of ambiguous command submissions.

Recovery owns no venue submission capability.  It can only reconcile the
prepared identity that was persisted before dispatch, and it keeps the cash
reservation when a venue cannot provide complete evidence of absence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Iterable, Optional

from .models import CommandState, TradeIntent
from .store import CommandClaim, ExecutionCommand, TwinStore, TwinStoreError


class RecoveryBlocked(RuntimeError):
    """Recovery did not have enough evidence to change durable exposure state."""


class RecoveryAction(str, Enum):
    TERMINAL = "terminal"
    RECONCILED = "reconciled"
    CONFIRMED_ABSENT = "confirmed_absent"
    OPERATOR_ATTENTION = "operator_attention"


@dataclass(frozen=True)
class VenueOrderLookup:
    """One complete, identity-bound order/fill query result."""

    account_scope_id: str
    instrument_id: str
    client_order_id: str
    request_fingerprint: str
    complete: bool
    order_found: Optional[bool]
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise RecoveryBlocked("recovery observation time must be timezone-aware")
        if not self.complete and self.order_found is not None:
            raise RecoveryBlocked("incomplete lookup cannot assert order presence or absence")
        if self.order_found is not None and not self.client_order_id:
            raise RecoveryBlocked("order lookup must retain the prepared client identity")


@dataclass(frozen=True)
class RecoveryResult:
    action: RecoveryAction
    command: ExecutionCommand
    reservation_released: bool = False


VenueLookup = Callable[[ExecutionCommand], VenueOrderLookup]


def _matching_lookup(command: ExecutionCommand, intent: TradeIntent, lookup: VenueOrderLookup) -> bool:
    return (
        lookup.complete
        and lookup.account_scope_id == command.scope_id
        and lookup.instrument_id == intent.instrument_id
        and lookup.client_order_id == command.client_order_id
        and lookup.request_fingerprint == command.request_fingerprint
    )


def recover_submission(
    store: TwinStore,
    *,
    command: ExecutionCommand,
    intent: TradeIntent,
    claim: CommandClaim,
    now: datetime,
    lookups: Iterable[VenueOrderLookup],
    required_absence_observations: int = 2,
) -> RecoveryResult:
    """Recover a command without resubmitting it.

    A confirmed order becomes acknowledged.  Releasing the reservation needs
    independent complete absence observations; any incomplete or mismatched
    observation is deliberately a paused operator state.
    """
    if required_absence_observations < 1:
        raise RecoveryBlocked("at least one complete absence observation is required")
    if now.tzinfo is None:
        raise RecoveryBlocked("recovery time must be timezone-aware")
    current = store.command_for_intent(intent)
    if current.id != command.id or current.intent_hash != intent.intent_hash:
        raise RecoveryBlocked("recovery command is not bound to the immutable intent")
    if current.state in {CommandState.FILLED, CommandState.CANCELLED, CommandState.REJECTED}:
        return RecoveryResult(RecoveryAction.TERMINAL, current)
    if current.state not in {CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN}:
        raise RecoveryBlocked(f"command cannot be recovered from {current.state.value}")
    if current.claim is None or current.claim.worker_id != claim.worker_id or current.claim.fence != claim.fence:
        raise RecoveryBlocked("recovery worker no longer owns the command fence")
    if current.claim.lease_expires_at <= now:
        raise RecoveryBlocked("recovery claim has expired")

    complete_absences = 0
    absence_observation_times: set[datetime] = set()
    for lookup in lookups:
        if lookup.observed_at > now or not _matching_lookup(current, intent, lookup):
            return RecoveryResult(RecoveryAction.OPERATOR_ATTENTION, current)
        if lookup.order_found is True:
            updated = store.transition_command(
                current.id, target=CommandState.ACKNOWLEDGED, fence=claim.fence, worker_id=claim.worker_id
            )
            return RecoveryResult(RecoveryAction.RECONCILED, updated)
        if lookup.order_found is False:
            if lookup.observed_at in absence_observation_times:
                return RecoveryResult(RecoveryAction.OPERATOR_ATTENTION, current)
            absence_observation_times.add(lookup.observed_at)
            complete_absences += 1

    if complete_absences < required_absence_observations:
        return RecoveryResult(RecoveryAction.OPERATOR_ATTENTION, current)
    updated = store.transition_command(
        current.id, target=CommandState.REJECTED, fence=claim.fence, worker_id=claim.worker_id
    )
    reservation_id = updated.reservation_id
    if store.durable:
        reservation_id = f"{updated.scope_id}:{reservation_id}"
    try:
        store.release_reservation(reservation_id, confirmed_no_order=True)
    except TwinStoreError as exc:
        raise RecoveryBlocked("confirmed absence could not release its reservation") from exc
    return RecoveryResult(RecoveryAction.CONFIRMED_ABSENT, updated, reservation_released=True)


def recovery_action(command_state: str, venue_order_found: bool | None) -> str:
    """Legacy simple classifier retained for callers without identity evidence."""
    if command_state in {"filled", "cancelled", "rejected"}:
        return "terminal"
    if command_state in {"submitting", "submission_unknown"}:
        if venue_order_found is True:
            return "reconcile"
        if venue_order_found is False:
            return "mark_no_order"
        return "pause_and_reconcile"
    return "resume"
