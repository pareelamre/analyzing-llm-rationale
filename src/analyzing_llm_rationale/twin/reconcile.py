"""Fail-closed cursor reconciliation for complete account generations.

This module deliberately keeps venue HTTP and credential handling outside the
state transition.  Adapters translate their pagination response to the small
``items``/``cursor`` contract below, then this reader either produces one
complete generation or retains the prior account snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

from .account import AccountSnapshot, AccountSyncResult, synchronize_account
from .models import SchemaValidationError


class AccountReadError(RuntimeError):
    """A venue read cannot safely become a complete account generation."""


PageFetcher = Callable[[Optional[str]], Mapping[str, Any]]


@dataclass(frozen=True)
class CompleteCollection:
    """All pages for one immutable account collection and its read fence."""

    name: str
    items: tuple[Mapping[str, Any], ...]
    generation_token: Optional[str]
    pages_read: int

    def account_pages(self) -> tuple[Mapping[str, Any], ...]:
        """Return the normalized single complete page consumed by account.py."""
        return ({"complete": True, "has_more": False, "items": list(self.items)},)


def read_complete_collection(
    name: str,
    fetch_page: PageFetcher,
    *,
    item_key: str = "items",
    cursor_key: str = "cursor",
    generation_key: str = "generation_token",
    max_pages: int = 1_000,
) -> CompleteCollection:
    """Read every opaque cursor page once, rejecting ambiguity rather than guessing.

    A non-empty generation token on the first response must remain unchanged.
    This lets an adapter fence a read using a venue snapshot/version marker. A
    venue without such a marker may still use the reader, but its adapter must
    provide a before/after fence before it labels the surrounding account read
    complete.
    """
    if not name.strip() or max_pages < 1:
        raise SchemaValidationError("collection name and positive page cap are required")
    cursor: Optional[str] = None
    seen_cursors: set[str] = set()
    rows: list[Mapping[str, Any]] = []
    generation_token: Optional[str] = None
    for page_number in range(1, max_pages + 1):
        try:
            response = fetch_page(cursor)
        except Exception as exc:
            raise AccountReadError(f"{name}_page_{page_number}_failed") from exc
        if not isinstance(response, Mapping):
            raise AccountReadError(f"{name}_page_{page_number}_malformed")
        value = response.get(item_key)
        if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
            raise AccountReadError(f"{name}_page_{page_number}_malformed")
        raw_generation = response.get(generation_key)
        current_generation = str(raw_generation).strip() if raw_generation not in (None, "") else None
        if page_number == 1:
            generation_token = current_generation
        elif current_generation != generation_token:
            raise AccountReadError(f"{name}_generation_changed")
        rows.extend(value)
        raw_cursor = response.get(cursor_key)
        next_cursor = str(raw_cursor).strip() if raw_cursor not in (None, "") else None
        if next_cursor is None:
            return CompleteCollection(name, tuple(rows), generation_token, page_number)
        if next_cursor in seen_cursors:
            raise AccountReadError(f"{name}_cursor_repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise AccountReadError(f"{name}_page_limit_exceeded")


def synchronize_complete_account(
    scope_id: str,
    *,
    generation: int,
    received_at: datetime,
    fetchers: Mapping[str, PageFetcher],
    local_command_ids: set[str],
    previous: Optional[AccountSnapshot] = None,
) -> AccountSyncResult:
    """Read all required collections before atomically replacing a snapshot.

    No partial collection is passed to ``synchronize_account``. A failed
    second page or one changed collection generation therefore retains the
    previous complete inventory and cannot increase spending capacity.
    """
    required = ("balances", "positions", "orders", "fills", "settlements")
    if any(name not in fetchers for name in required):
        raise SchemaValidationError("complete account synchronization needs every collection fetcher")
    collections: dict[str, CompleteCollection] = {}
    issues: list[str] = []
    for name in required:
        try:
            collections[name] = read_complete_collection(name, fetchers[name])
        except AccountReadError as exc:
            issues.append(str(exc))
    if issues:
        return AccountSyncResult(previous, previous is not None, tuple(issues))
    return synchronize_account(
        scope_id,
        generation=generation,
        received_at=received_at,
        balances=collections["balances"].account_pages(),
        positions=collections["positions"].account_pages(),
        orders=collections["orders"].account_pages(),
        fills=collections["fills"].account_pages(),
        settlements=collections["settlements"].account_pages(),
        local_command_ids=local_command_ids,
        previous=previous,
    )
