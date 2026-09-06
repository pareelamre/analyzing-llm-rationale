"""Causal replay helpers: observed-at cutoff precedes every decision."""
from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence


def causal_events(events: Sequence[Mapping], *, as_of: datetime) -> list[Mapping]:
    """Deduplicate stable IDs and exclude later-observed information."""
    selected = {}
    for event in events:
        observed = event.get("observed_at")
        event_id = event.get("id")
        if not event_id or not isinstance(observed, datetime) or observed.tzinfo is None or observed > as_of:
            continue
        selected.setdefault(str(event_id), event)
    return [selected[key] for key in sorted(selected)]
