"""Causal replay helpers: observed-at and occurred-at both precede decisions."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping, Sequence


class ReplayValidationError(ValueError):
    """Captured replay input is ambiguous, future-dated, or malformed."""


def _timestamp(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) and value.tzinfo is not None else None


def _canonical(event: Mapping[str, Any]) -> str:
    return json.dumps(dict(event), sort_keys=True, default=str, separators=(",", ":"))


def causal_events(events: Sequence[Mapping], *, as_of: datetime) -> list[Mapping]:
    """Return one immutable fact per ID available at the frozen decision time."""
    if as_of.tzinfo is None:
        raise ReplayValidationError("as_of must be timezone-aware")
    selected: dict[str, Mapping] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise ReplayValidationError("replay event must be an object")
        observed, occurred = _timestamp(event.get("observed_at")), _timestamp(event.get("occurred_at", event.get("observed_at")))
        event_id = str(event.get("id") or "").strip()
        if not event_id or observed is None or occurred is None:
            raise ReplayValidationError("replay events require ID, observed_at, and occurred_at")
        if observed > as_of or occurred > as_of:
            continue
        prior = selected.get(event_id)
        if prior is not None and _canonical(prior) != _canonical(event):
            raise ReplayValidationError(f"conflicting replay event ID: {event_id}")
        selected[event_id] = event
    return [selected[key] for key in sorted(selected)]