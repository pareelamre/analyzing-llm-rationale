"""Private, authority-fenced continuation of one native order history capture."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Callable, Mapping

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .models import AccountScope
from .reconcile import AccountReadError, read_complete_collection
from .store import ExecutionCommand

tracer = trace.get_tracer(__name__)
pages = metrics.get_meter(__name__).create_counter("twin.native_history.pages", unit="1")
collections = metrics.get_meter(__name__).create_counter(
    "twin.native_history.collections", unit="1"
)
_MAX_BYTES = 900_000
_MAX_PAGES = 4096
_MAX_ROWS = 4096


class NativeHistoryPending(AccountReadError):
    """A complete bounded read is unfinished; its saved cursor remains authoritative."""


def _json(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode()) > _MAX_BYTES:
            raise ValueError("capture too large")
        return encoded
    except (TypeError, ValueError):
        raise AccountReadError("native_history_size_or_shape_invalid") from None


def _cursor(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise AccountReadError("native_history_cursor_invalid")
    return value


class NativeHistoryCapture:
    """One coherent durable client, one command, at most eight pages per default job.

    Only selected rows are retained. No private history is indexed, logged or
    published as inventory. Call clear_in_transaction only with lifecycle state.
    """

    def __init__(
        self,
        client: Any,
        scope: AccountScope,
        command: ExecutionCommand,
        *,
        guard: Callable[[], None],
        clock: Callable[[], datetime],
        deadline: datetime,
        page_budget: int = 8,
    ):
        if type(page_budget) is not int or not 1 <= page_budget <= 32:
            raise ValueError("native history page budget must be between1 and32")
        self.client, self.guard, self.clock, self.deadline = client, guard, clock, deadline
        self.remaining = page_budget
        self.context = sha256(
            _json(
                {
                    "scope": scope.id,
                    "epoch": scope.account_epoch,
                    "venue": scope.venue,
                    "environment": scope.environment,
                    "account": scope.venue_account_ref,
                    "connection": scope.connection_ref,
                    "command": command.id,
                    "intent": command.intent_hash,
                    "request": command.request_fingerprint,
                }
            ).encode()
        ).hexdigest()
        self.root_key = client.key("TwinNativeHistory", self.context)
        root = self._root()
        self.frame, self.order_id = root["frame"], root["order_id"]
        if root.get("discard_on_resume") is True:
            with self.client.transaction():
                root = self._root()
                if root.get("discard_on_resume") is True:
                    for query in root["collections"]:
                        self.client.delete(self._chunk_key(query))
                    root.update(collections=[], discard_on_resume=False)
                    self._put(self.root_key, root)
                    self.guard()

    def _load(self, key):
        entity = self.client.get(key)
        if entity is None:
            return None
        raw = entity.get("payload_json")
        try:
            if not isinstance(raw, str) or len(raw.encode()) > _MAX_BYTES:
                raise ValueError("invalid size")
            if sha256(raw.encode()).hexdigest() != entity.get("payload_sha256"):
                raise ValueError("digest changed")
            value = json.loads(raw)
            if value["context"] != self.context or value["schema_version"] != 1:
                raise ValueError("context changed")
            return value
        except (KeyError, TypeError, ValueError):
            raise AccountReadError("native_history_corrupt") from None

    def _put(self, key, payload):
        from .. import datastore_backend as datastore

        raw = _json({**payload, "context": self.context, "schema_version": 1})
        entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
        entity.update(payload_json=raw, payload_sha256=sha256(raw.encode()).hexdigest())
        self.client.put(entity)

    def _root(self):
        self.guard()
        root = self._load(self.root_key)
        if root is None:
            return {"frame": "discovery", "order_id": None, "collections": []}
        try:
            if (
                not isinstance(root["frame"], str)
                or not isinstance(root["collections"], list)
                or len(root["collections"]) > 8
                or len(set(root["collections"])) != len(root["collections"])
                or any(
                    not isinstance(x, str)
                    or len(x) != 64
                    or any(c not in "0123456789abcdef" for c in x)
                    for x in root["collections"]
                )
                or (
                    root["order_id"] is not None
                    and (
                        not isinstance(root["order_id"], str)
                        or not root["order_id"]
                        or len(root["order_id"]) > 200
                    )
                )
            ):
                raise ValueError("invalid root")
        except (KeyError, TypeError, ValueError):
            raise AccountReadError("native_history_manifest_invalid") from None
        return root

    def _chunk_key(self, query):
        return self.client.key("TwinNativeHistoryChunk", self.context + ":" + query)

    def select_order(self, order_id: str, frame: Mapping[str, Any]):
        fingerprint = sha256(_json(frame).encode()).hexdigest()
        with self.client.transaction():
            root = self._root()
            if root["order_id"] not in (None, order_id):
                raise AccountReadError("native_history_order_changed")
            if root["frame"] != fingerprint:
                for query in root["collections"]:
                    self.client.delete(self._chunk_key(query))
                root = {"frame": fingerprint, "order_id": order_id, "collections": []}
            self._put(self.root_key, root)
            self.guard()
        self.frame, self.order_id = fingerprint, order_id

    @tracer.start_as_current_span(
        "twin.native_history.collection", record_exception=False, set_status_on_exception=False
    )
    def collection(self, name, parameters, fetch, *, select=lambda row: True):
        try:
            result = self._collection(name, parameters, fetch, select=select)
            collections.add(1, {"outcome": "complete"})
            return result
        except NativeHistoryPending:
            collections.add(1, {"outcome": "pending"})
            raise
        # aqg: top-level boundary -- never export SDK text or private capture rows
        except Exception:
            collections.add(1, {"outcome": "failed"})
            trace.get_current_span().set_status(
                Status(StatusCode.ERROR, "native_history_unconfirmed")
            )
            raise

    def _collection(self, name, parameters, fetch, *, select=lambda row: True):
        if name not in {
            "orders",
            "historical_orders",
            "fills",
            "historical_fills",
            "trades_by_market",
        }:
            raise AccountReadError("native_history_collection_invalid")
        query = sha256(_json([name, parameters]).encode()).hexdigest()
        key = self._chunk_key(query)
        root = self._root()
        if root["frame"] != self.frame:
            raise AccountReadError("native_history_frame_changed")
        saved = self._load(key)
        if saved is not None and (
            query not in root["collections"] or saved.get("frame") != self.frame
        ):
            raise AccountReadError("native_history_frame_changed")
        state = saved or {
            "frame": self.frame,
            "page_count": 0,
            "cursor": None,
            "seen": [],
            "rows": [],
            "generation": None,
            "complete": False,
        }
        try:
            if (
                type(state["page_count"]) is not int
                or not 0 <= state["page_count"] <= _MAX_PAGES
                or not isinstance(state["seen"], list)
                or len(state["seen"]) > _MAX_PAGES
                or len(set(state["seen"])) != len(state["seen"])
                or any(not isinstance(x, str) or len(x) != 64 for x in state["seen"])
                or not isinstance(state["rows"], list)
                or len(state["rows"]) > _MAX_ROWS
                or any(not isinstance(x, dict) for x in state["rows"])
                or type(state["complete"]) is not bool
                or (state["complete"] and state["cursor"] is not None)
            ):
                raise ValueError("invalid state")
            _cursor(state["cursor"])
        except (KeyError, TypeError, ValueError):
            raise AccountReadError("native_history_state_invalid") from None
        while not state["complete"]:
            self.guard()
            if self.remaining == 0 or self.clock() + timedelta(seconds=5) >= self.deadline:
                trace.get_current_span().set_attribute("outcome", "pending")
                raise NativeHistoryPending("native_history_incomplete")
            if state["page_count"] >= _MAX_PAGES:
                raise AccountReadError("native_history_page_limit_reached")
            self.remaining -= 1
            response = fetch(state["cursor"])
            self.guard()
            if not isinstance(response, Mapping) or "cursor" not in response:
                raise AccountReadError("native_history_page_incomplete")
            next_cursor = _cursor(response["cursor"])
            # Reuse the existing collection shape/generation/truncation checks.
            page = read_complete_collection(
                name, lambda _, response=response: {**response, "cursor": None}, max_pages=1
            )
            if state["page_count"] and page.generation_token != state["generation"]:
                raise AccountReadError("native_history_generation_changed")
            seen = list(state["seen"])
            if next_cursor is not None:
                digest = sha256(next_cursor.encode()).hexdigest()
                if digest in seen:
                    raise AccountReadError("native_history_cursor_repeated")
                seen.append(digest)
            rows = [*state["rows"], *(dict(row) for row in page.items if select(row))]
            if len(rows) > _MAX_ROWS:
                raise AccountReadError("native_history_row_limit_reached")
            updated = {
                "frame": self.frame,
                "page_count": state["page_count"] + 1,
                "cursor": next_cursor,
                "seen": seen,
                "rows": rows,
                "generation": page.generation_token,
                "complete": next_cursor is None,
            }
            with self.client.transaction():
                current_root = self._root()
                current = self._load(key)
                if current_root["frame"] != self.frame or current != saved:
                    raise AccountReadError("native_history_concurrent_change")
                if query not in current_root["collections"]:
                    current_root["collections"].append(query)
                    if len(current_root["collections"]) > 8:
                        raise AccountReadError("native_history_collection_limit_reached")
                self._put(self.root_key, current_root)
                self._put(key, updated)
                self.guard()
            state = updated
            saved = {**updated, "context": self.context, "schema_version": 1}
            pages.add(1, {"outcome": "saved"})
        trace.get_current_span().set_attribute("outcome", "complete")
        return tuple(state["rows"])

    def invalidate_complete_on_error(self):
        """Refresh rejected complete cohorts; retain partial cursor progress on outages."""
        with self.client.transaction():
            root = self._root()
            if root["frame"] != self.frame:
                raise AccountReadError("native_history_frame_changed")
            complete = bool(root["collections"]) and all(
                (self._load(self._chunk_key(query)) or {}).get("complete") is True
                for query in root["collections"]
            )
            if complete:
                root["discard_on_resume"] = True
                self._put(self.root_key, root)
                self.guard()

    def clear_in_transaction(self):
        """Join the lifecycle/continuation transaction; no nested Datastore commit."""
        root = self._root()
        if root["frame"] != self.frame:
            raise AccountReadError("native_history_frame_changed")
        for query in root["collections"]:
            self.client.delete(self._chunk_key(query))
        self.client.delete(self.root_key)
        self.guard()
