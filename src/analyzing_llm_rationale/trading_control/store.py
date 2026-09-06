"""Store interfaces and transitions used by trading execution callers.

The caller owns the transaction or process lock.  This module deliberately has
no Datastore or FastAPI import so workers can supply their durable store without
booting the HTTP application.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

from .runs import claim_for_submission


class SavedRunStore(Protocol):
    """Minimal mutable record capability required to claim one saved run."""

    def read(self) -> Optional[Mapping[str, Any]]:
        """Return the current run record in the caller's transaction."""

    def write(self, record: Mapping[str, Any]) -> None:
        """Persist the claimed record in the caller's transaction."""


@dataclass(frozen=True)
class CallbackSavedRunStore:
    """Adapter for an application's existing read/write primitives."""

    load: Callable[[], Optional[Mapping[str, Any]]]
    save: Callable[[Mapping[str, Any]], None]

    def read(self) -> Optional[Mapping[str, Any]]:
        return self.load()

    def write(self, record: Mapping[str, Any]) -> None:
        self.save(record)


def claim_saved_run(
    store: SavedRunStore,
    preview: Mapping[str, Any],
    *,
    clock: Callable[[], str],
) -> tuple[Optional[dict[str, Any]], bool]:
    """Claim an awaiting run using caller-injected durable storage and time.

    This function is intentionally not a locking primitive.  A production
    implementation calls it inside a Datastore transaction, while local tests
    hold their own lock.  Keeping that choice at the boundary prevents worker
    code from depending on server globals.
    """
    record = store.read()
    if record is None:
        return None, False
    claimed, acquired = claim_for_submission(record, preview, approved_at=clock())
    if acquired:
        store.write(claimed)
    return claimed, acquired
