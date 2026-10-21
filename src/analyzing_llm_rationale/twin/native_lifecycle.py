"""Positive native order/fill observations applied through fenced private recovery.

This module has no order submission or cancellation capability. Missing reads
are unknown. Market-level redemptions are never allocated to an individual order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Callable, Mapping

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .. import trading, venue_api
from .models import AccountScope, CommandState, TradeIntent, canonical_instrument_id
from .native_history import NativeHistoryCapture, NativeHistoryPending
from .reconcile import AccountReadError, cursor_page_fetcher, read_complete_collection
from .recovery import (
    DatastoreLifecycleStore,
    FillObservation,
    VenueOrderLookup,
    reconcile_lifecycle,
    recover_submission,
)
from .store import DatastoreTwinStore, ExecutionCommand, command_request_fingerprint
from .venue_account import complete_account_read_plan
from .worker import DatastoreWorkerJobs, WorkerDegraded, WorkerJob, WorkerJobKind, WorkerJobStatus

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
operations = metrics.get_meter(__name__).create_counter(
    "twin.native_lifecycle.operations", unit="1"
)


def _number(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
        if isinstance(value, bool) or not result.is_finite() or result < 0:
            raise ValueError("invalid native number")
        return result
    except (ValueError, TypeError, InvalidOperation):
        raise AccountReadError("native_economic_field_unavailable") from None


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise AccountReadError("native_identity_unavailable")
    return value


def _timestamp(value: Any) -> datetime:
    try:
        if isinstance(value, str) and "T" in value:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            result = datetime.fromtimestamp(float(_number(value)), timezone.utc)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError("naive native timestamp")
        return result
    except (ValueError, TypeError, OverflowError, OSError):
        raise AccountReadError("native_timestamp_unavailable") from None


def _fee(row: Mapping[str, Any]) -> Decimal:
    for name in ("fee", "fee_usdc", "fee_cost"):
        if name in row:
            return _number(row[name])
    if "fee_rate_bps" in row and _number(row["fee_rate_bps"]) == 0:
        return Decimal("0")
    raise AccountReadError("native_actual_fee_unavailable")


@dataclass(frozen=True)
class NativeOrderObservation:
    scope_id: str
    account_epoch: int
    command_id: str
    intent_hash: str
    venue_order_id: str
    order_status: str
    fills: tuple[FillObservation, ...]
    observed_at: datetime
    terminal_evidence_complete: bool
    pending_trades: int = 0


class NativeOrderReader:
    """Read one prepared order and every page of its native fill evidence."""

    def __init__(
        self,
        scope: AccountScope,
        *,
        creds: Mapping[str, Any],
        connection_ref: str,
        reader: Callable[..., Mapping[str, Any]] = venue_api.read,
    ):
        self.scope, self.creds, self.connection_ref = scope, dict(creds), connection_ref
        self.reader = reader

    @tracer.start_as_current_span(
        "twin.native_lifecycle.capture", record_exception=False, set_status_on_exception=False
    )
    def capture(
        self,
        command: ExecutionCommand,
        intent: TradeIntent,
        *,
        clock: Callable[[], datetime],
        guard: Callable[[], None],
        known_order_id: str | None = None,
        history: NativeHistoryCapture | None = None,
    ) -> NativeOrderObservation:
        try:
            result = self._capture(
                command,
                intent,
                clock=clock,
                guard=guard,
                known_order_id=known_order_id,
                history=history,
            )
            trace.get_current_span().set_attributes(
                {
                    "venue": self.scope.venue,
                    "fills.count": len(result.fills),
                    "outcome": "complete" if result.terminal_evidence_complete else "pending",
                }
            )
            return result
        except NativeHistoryPending:
            trace.get_current_span().set_attribute("outcome", "history_pending")
            raise
        # aqg: top-level boundary -- fixed status only, never raw SDK exceptions
        except BaseException:
            trace.get_current_span().set_status(
                Status(StatusCode.ERROR, "native_capture_unconfirmed")
            )
            raise

    def _capture(
        self,
        command: ExecutionCommand,
        intent: TradeIntent,
        *,
        clock: Callable[[], datetime],
        guard: Callable[[], None],
        known_order_id: str | None = None,
        history: NativeHistoryCapture | None = None,
    ) -> NativeOrderObservation:
        read_creds = dict(self.creds)

        def wait_for_coherent_history(reason):
            if history is None:
                raise AccountReadError(reason)
            with history.client.transaction():
                history.clear_in_transaction()
            raise NativeHistoryPending(reason)

        def read(operation, parameters):
            guard()
            result = self.reader(
                self.scope.venue, operation, parameters, access="account", creds=dict(read_creds)
            )
            guard()
            if not isinstance(result, Mapping) or not isinstance(result.get("data"), Mapping):
                raise AccountReadError("native_response_malformed")
            return result

        def collection(operation, key, parameters, select=lambda row: True):
            def read_page(_venue, operation, parameters, **_kwargs):
                result = read(operation, parameters)
                if "next_cursor" not in result:
                    raise AccountReadError("native_collection_page_incomplete")
                if self.scope.venue == "polymarket":
                    if "next_cursor" not in result["data"]:
                        raise AccountReadError("native_trade_page_incomplete")
                    raw_cursor = result["data"]["next_cursor"]
                    expected = None if raw_cursor in ("", "LTE=") else raw_cursor
                    if result["next_cursor"] != expected:
                        raise AccountReadError("native_trade_cursor_mismatch")
                return result

            fetch = cursor_page_fetcher(
                self.scope.venue,
                operation,
                item_key=key,
                reader=read_page,
                creds=self.creds,
                parameters=parameters,
                cursor_parameter="next_cursor" if self.scope.venue == "polymarket" else "cursor",
            )
            if history is not None:
                return history.collection(operation, parameters, fetch, select=select)
            return read_complete_collection(key, fetch, max_pages=20).items

        guard()
        if (
            self.scope.environment != "live"
            or command.scope_id != self.scope.id
            or intent.account_scope_id != self.scope.id
            or intent.account_epoch != self.scope.account_epoch
            or command.intent_hash != intent.intent_hash
            or command.intent_id != intent.id
            or command.request_fingerprint
            != command_request_fingerprint(
                intent, command.client_order_id, command.prepared_request_json
            )
        ):
            raise AccountReadError("native_command_binding_changed")
        plan = complete_account_read_plan(
            self.scope.venue,
            reader=self.reader,
            creds=self.creds,
            connection_ref=self.connection_ref,
            read_guard=guard,
        )
        if plan.account_binding != (
            self.scope.connection_ref,
            self.scope.venue_account_ref.lower(),
        ):
            raise AccountReadError("native_account_binding_changed")
        if self.scope.venue == "polymarket":
            read_creds["polymarket_funder_address"] = self.scope.venue_account_ref
        guard()
        if self.scope.venue == "kalshi":
            ticker = intent.instrument_id.split(":", 2)[-1]
            history_min_ts = max(0, int(intent.created_at.timestamp()) - 1)
            if (
                canonical_instrument_id(
                    venue="kalshi", environment="live", venue_instrument_id=ticker
                )
                != intent.instrument_id
            ):
                raise AccountReadError("native_instrument_binding_changed")

            def order(order_id=None):
                if order_id is not None:
                    row = read("order", {"order_id": order_id})["data"].get("order")
                else:
                    rows = (
                        *collection(
                            "orders",
                            "orders",
                            {"ticker": ticker, "limit": 100, "subaccount": 0},
                            select=lambda row: (
                                row.get("client_order_id") == command.client_order_id
                            ),
                        ),
                        *collection(
                            "historical_orders",
                            "orders",
                            {
                                "ticker": ticker,
                                "limit": 100,
                                "subaccount": 0,
                                "min_ts": history_min_ts,
                            },
                            select=lambda row: (
                                row.get("client_order_id") == command.client_order_id
                            ),
                        ),
                    )
                    matches = [
                        row for row in rows if row.get("client_order_id") == command.client_order_id
                    ]
                    unique = {str(row.get("order_id")): row for row in matches}
                    if len(unique) != 1 or any(
                        row != next(iter(unique.values())) for row in matches
                    ):
                        raise AccountReadError("native_order_presence_unconfirmed")
                    row = next(iter(unique.values()))
                if not isinstance(row, Mapping):
                    raise AccountReadError("native_order_presence_unconfirmed")
                if order_id is not None and row.get("order_id") != order_id:
                    raise AccountReadError("native_order_identity_mismatch")
                expected_side = "yes" if intent.action.value.endswith("_YES") else "no"
                expected_action = "buy" if intent.action.value.startswith("BUY_") else "sell"
                if (
                    row.get("client_order_id") != command.client_order_id
                    or row.get("ticker") != ticker
                    or row.get("side") != expected_side
                    or row.get("action") != expected_action
                    or type(row.get("subaccount_number")) is not int
                    or row["subaccount_number"] != 0
                    or _number(row.get("initial_count_fp")) != intent.quantity
                    or _number(row.get(expected_side + "_price_dollars")) != intent.limit_price
                ):
                    raise AccountReadError("native_order_identity_mismatch")
                return dict(row)

            previous_id = known_order_id or (history.order_id if history is not None else None)
            first = order(previous_id)
            if history is not None and previous_id is None:
                first = order(_identifier(first.get("order_id")))
            native_id = _identifier(first.get("order_id"))
            if history is not None:
                history.select_order(
                    native_id,
                    {
                        "status": first.get("status"),
                        "filled": str(_number(first.get("fill_count_fp")).normalize()),
                        "remaining": str(_number(first.get("remaining_count_fp")).normalize()),
                    },
                )
            raw_fills = (
                *collection(
                    "fills",
                    "fills",
                    {"ticker": ticker, "order_id": native_id, "limit": 100, "subaccount": 0},
                    select=lambda row: _identifier(row.get("order_id")) == native_id,
                ),
                *collection(
                    "historical_fills",
                    "fills",
                    {"ticker": ticker, "limit": 100, "subaccount": 0, "min_ts": history_min_ts},
                    select=lambda row: _identifier(row.get("order_id")) == native_id,
                ),
            )
            economic = []
            for row in raw_fills:
                _identifier(row.get("order_id"))
                if row.get("order_id") != native_id:
                    continue
                side = "yes" if intent.action.value.endswith("_YES") else "no"
                action = "buy" if intent.action.value.startswith("BUY_") else "sell"
                if (
                    row.get("ticker") != ticker
                    or row.get("side") != side
                    or row.get("action") != action
                    or type(row.get("subaccount_number")) is not int
                    or row["subaccount_number"] != 0
                ):
                    raise AccountReadError("native_fill_identity_mismatch")
                economic.append(
                    (
                        _identifier(row.get("fill_id")),
                        _number(row.get("count_fp")),
                        _number(row.get(side + "_price_dollars")),
                        _fee(row),
                        _timestamp(row.get("created_time")),
                    )
                )
            second = order(native_id)

            def frame(row):
                return (
                    row.get("order_id"),
                    row.get("status"),
                    row.get("fill_count_fp"),
                    row.get("remaining_count_fp"),
                )

            reported = _number(first.get("fill_count_fp"))
            remaining = _number(first.get("remaining_count_fp"))
            if remaining + reported > intent.quantity or (
                first.get("status") == "resting" and remaining + reported != intent.quantity
            ):
                raise AccountReadError("native_order_quantity_incoherent")
            status = {
                "resting": "open",
                "executed": "filled",
                "canceled": "cancelled",
                "cancelled": "cancelled",
            }.get(first.get("status"))
            pending = 0
            created = _timestamp(first.get("created_time"))
            creation_field = "created_time"
        elif self.scope.venue == "polymarket":
            try:
                packet = json.loads(command.prepared_request_json)
                wire = packet["order"]
                expected_side = "BUY" if intent.action.value.startswith("BUY_") else "SELL"
                shares, cash = (
                    intent.quantity * 1_000_000,
                    intent.quantity * intent.limit_price * 1_000_000,
                )
                amounts = (cash, shares) if expected_side == "BUY" else (shares, cash)
                if (
                    packet["protocol"] != "ctf_v2"
                    or packet["chain_id"] != 137
                    or packet["intent_hash"] != intent.intent_hash
                    or canonical_instrument_id(
                        venue="polymarket",
                        environment="live",
                        venue_instrument_id=packet["venue_instrument_id"],
                        condition_id=packet["condition_id"],
                    )
                    != intent.instrument_id
                    or wire["side"] != expected_side
                    or str(wire["maker"]).lower() != self.scope.venue_account_ref.lower()
                    or (_number(wire["makerAmount"]), _number(wire["takerAmount"])) != amounts
                ):
                    raise ValueError("prepared context mismatch")
                token = _identifier(wire["tokenId"])
            except (KeyError, TypeError, ValueError):
                raise AccountReadError("native_prepared_context_unavailable") from None
            guard()
            client = trading._polymarket_client(self.creds)
            guard()
            if (
                str(client.builder.funder).lower() != self.scope.venue_account_ref.lower()
                or client.host != "https://clob.polymarket.com"
            ):
                raise AccountReadError("native_client_binding_changed")

            def order(_order_id=None):
                row = read("order", {"order_id": command.client_order_id})["data"]
                if (
                    row.get("id") != command.client_order_id
                    or row.get("market") != packet["condition_id"]
                    or row.get("asset_id") != token
                    or row.get("side") != expected_side
                    or str(row.get("maker_address") or "").lower()
                    != self.scope.venue_account_ref.lower()
                    or _number(row.get("price")) != intent.limit_price
                    or _number(row.get("original_size")) != intent.quantity
                ):
                    raise AccountReadError("native_order_identity_mismatch")
                return dict(row)

            first = order()
            native_id = command.client_order_id
            trade_ids = first.get("associate_trades")
            if (
                not isinstance(trade_ids, list)
                or len(trade_ids) > 1000
                or any(not isinstance(x, str) or not x for x in trade_ids)
                or len(set(trade_ids)) != len(trade_ids)
            ):
                raise AccountReadError("native_trade_inventory_unavailable")
            if history is not None:
                history.select_order(
                    native_id,
                    {
                        "status": first.get("status"),
                        "filled": str(_number(first.get("size_matched")).normalize()),
                        "trades": sorted(trade_ids),
                    },
                )
            wanted = set(trade_ids)

            def belongs_to_order(row):
                trade_id = _identifier(row.get("id"))
                makers = row.get("maker_orders", [])
                if not isinstance(makers, list) or any(
                    not isinstance(leg, Mapping) for leg in makers
                ):
                    raise AccountReadError("native_maker_legs_malformed")
                return (
                    trade_id in wanted
                    or row.get("taker_order_id") == native_id
                    or any(leg.get("order_id") == native_id for leg in makers)
                )

            rows = collection(
                "fills",
                "data",
                {
                    "market": packet["condition_id"],
                    "after": str(max(0, int(intent.created_at.timestamp()) - 1)),
                },
                select=belongs_to_order,
            )
            by_id = {}
            for row in rows:
                trade_id = _identifier(row.get("id"))
                if trade_id not in wanted:
                    wait_for_coherent_history("native_trade_inventory_incoherent")
                if trade_id in by_id and by_id[trade_id] != row:
                    raise AccountReadError("native_trade_revision_conflict")
                by_id[trade_id] = row
            if set(by_id) != wanted:
                wait_for_coherent_history("native_associated_trade_missing")
            economic, pending = [], 0
            for trade_id, row in by_id.items():
                if row.get("market") != packet["condition_id"]:
                    raise AccountReadError("native_trade_identity_mismatch")
                makers = row.get("maker_orders", [])
                if not isinstance(makers, list) or any(
                    not isinstance(leg, Mapping) for leg in makers
                ):
                    raise AccountReadError("native_maker_legs_malformed")
                legs = [leg for leg in makers if leg.get("order_id") == native_id]
                if row.get("taker_order_id") == native_id:
                    if legs or row.get("trader_side") != "TAKER":
                        raise AccountReadError("native_trade_role_mismatch")
                    leg, size = row, row.get("size")
                    is_maker = False
                    if makers:
                        ids = [_identifier(item.get("order_id")) for item in makers]
                        amounts = [_number(item.get("matched_amount")) for item in makers]
                        if (
                            len(set(ids)) != len(ids)
                            or any(amount <= 0 for amount in amounts)
                            or sum(amounts) != _number(size)
                        ):
                            raise AccountReadError("native_taker_maker_quantity_mismatch")
                        if all(item.get("asset_id") == token for item in makers):
                            opposite = "SELL" if expected_side == "BUY" else "BUY"
                            prices = [_number(item.get("price")) for item in makers]
                            if (
                                any(item.get("side") != opposite for item in makers)
                                or any(price > 1 for price in prices)
                                or sum(amount * price for amount, price in zip(amounts, prices))
                                != _number(size) * _number(row.get("price"))
                            ):
                                raise AccountReadError("native_taker_maker_notional_mismatch")
                else:
                    if len(legs) != 1 or row.get("trader_side") != "MAKER":
                        raise AccountReadError("native_trade_role_mismatch")
                    leg, size = legs[0], legs[0].get("matched_amount")
                    is_maker = True
                if (
                    leg.get("asset_id") != token
                    or leg.get("side") != expected_side
                    or (
                        is_maker
                        and str(leg.get("maker_address") or "").lower()
                        != self.scope.venue_account_ref.lower()
                    )
                ):
                    raise AccountReadError("native_trade_leg_identity_mismatch")
                trade_status = str(row.get("status") or "").removeprefix("TRADE_STATUS_")
                if trade_status in {"MATCHED", "MINED", "RETRYING"}:
                    pending += 1
                    continue
                if trade_status != "CONFIRMED":
                    raise AccountReadError("native_trade_finality_unconfirmed")
                economic.append(
                    (
                        "polymarket:" + trade_id + ":" + native_id,
                        _number(size),
                        _number(leg.get("price")),
                        _fee(leg),
                        _timestamp(row.get("match_time")),
                    )
                )
            second = order()

            def frame(row):
                return (
                    row.get("id"),
                    row.get("status"),
                    row.get("size_matched"),
                    tuple(row.get("associate_trades", [])),
                )

            reported = _number(first.get("size_matched"))
            raw_status = str(first.get("status") or "").removeprefix("ORDER_STATUS_")
            status = {
                "LIVE": "open",
                "UNMATCHED": "open",
                "MATCHED": "filled",
                "DELAYED": "acknowledged",
                "CANCELED": "cancelled",
                "CANCELLED": "cancelled",
            }.get(raw_status)
            created = _timestamp(first.get("created_at"))
            creation_field = "created_at"
        else:
            raise AccountReadError("native_venue_unsupported")
        guard()
        observed = clock()
        if frame(first) != frame(second):
            wait_for_coherent_history("native_order_changed_during_capture")
        if (
            status is None
            or reported > intent.quantity
            or _timestamp(second.get(creation_field)) != created
        ):
            raise AccountReadError("native_order_changed_during_capture")
        if created < intent.created_at.replace(microsecond=0) or created > observed:
            raise AccountReadError("native_order_creation_mismatch")
        unique = {}
        for fill_id, quantity, price, fee, occurred in economic:
            if (
                price > 1
                or (intent.action.value.startswith("BUY_") and price > intent.limit_price)
                or (intent.action.value.startswith("SELL_") and price < intent.limit_price)
            ):
                raise AccountReadError("native_fill_price_mismatch")
            fill = FillObservation(
                fill_id,
                1,
                native_id,
                command.client_order_id,
                intent.instrument_id,
                quantity,
                occurred,
                observed,
                price=price,
                fee=fee,
            )
            if fill_id in unique and unique[fill_id] != fill:
                raise AccountReadError("native_fill_revision_conflict")
            unique[fill_id] = fill
        fills = tuple(unique[key] for key in sorted(unique))
        total = sum((fill.quantity for fill in fills), Decimal("0"))
        if total > reported or (not pending and total != reported):
            raise AccountReadError("native_fill_quantity_incomplete")
        terminal = not pending and status in {"filled", "cancelled"} and total == reported
        if (pending and status in {"filled", "cancelled"}) or (
            status == "filled" and total != intent.quantity
        ):
            status = "partially_filled" if total else "acknowledged"
            terminal = False
        return NativeOrderObservation(
            self.scope.id,
            self.scope.account_epoch,
            command.id,
            intent.intent_hash,
            native_id,
            status,
            fills,
            observed,
            terminal,
            pending,
        )


class NativeLifecycleMaintenance:
    """One claimed private job; all provider work occurs outside financial commits."""

    def __init__(
        self,
        store: DatastoreTwinStore,
        lifecycle: DatastoreLifecycleStore,
        jobs: DatastoreWorkerJobs,
        *,
        resolve_reader: Callable[[AccountScope], NativeOrderReader],
        clock: Callable[[], datetime],
        dispatcher: Any | None = None,
    ):
        if (
            not isinstance(store, DatastoreTwinStore)
            or not isinstance(lifecycle, DatastoreLifecycleStore)
            or not isinstance(jobs, DatastoreWorkerJobs)
            or store._client is not lifecycle._client
            or store._client is not jobs._client
        ):
            raise ValueError("native lifecycle requires one coherent durable client")
        self.store, self.lifecycle, self.jobs, self.resolve_reader, self.clock = (
            store,
            lifecycle,
            jobs,
            resolve_reader,
            clock,
        )
        self.dispatcher = dispatcher

    def _dispatch_followup(self, followup):
        if followup is not None and self.dispatcher is not None:
            try:
                self.dispatcher.enqueue(followup)
            except Exception:  # aqg: top-level boundary -- durable queue retains lost dispatches
                operations.add(1, {"outcome": "followup_dispatch_pending"})
                logger.warning("Native recovery follow-up remains queued")

    @tracer.start_as_current_span(
        "twin.native_lifecycle.maintain", record_exception=False, set_status_on_exception=False
    )
    def __call__(self, job: WorkerJob) -> dict[str, object]:
        history = None
        try:
            started = self.clock()
            basic = {"command_id", "account_epoch_id"}
            if job.kind is not WorkerJobKind.RECOVERY or set(job.payload) not in (
                basic,
                basic | {"recovery_root_id", "recovery_round_id"},
            ):
                raise WorkerDegraded("native_lifecycle_job_invalid")
            round_id = job.payload.get("recovery_round_id", "0")
            if not round_id.isdigit() or str(int(round_id)) != round_id or int(round_id) > 16:
                raise WorkerDegraded("native_lifecycle_job_invalid")
            recovery_round = int(round_id)
            root_id = job.payload.get("recovery_root_id", job.id)
            scope = self.store.account_scope(job.account_scope_id)
            if (
                job.payload["account_epoch_id"] != str(scope.account_epoch)
                or scope.environment != "live"
            ):
                raise WorkerDegraded("native_lifecycle_scope_changed")
            intent = self.store.intent_for_command(scope.id, job.payload["command_id"])
            command = self.store.command_for_intent(intent)
            if (
                command.id != job.payload["command_id"]
                or intent.account_epoch != scope.account_epoch
            ):
                raise WorkerDegraded("native_lifecycle_command_changed")
            if command.state not in {
                CommandState.SUBMITTING,
                CommandState.SUBMISSION_UNKNOWN,
                CommandState.ACKNOWLEDGED,
                CommandState.PARTIALLY_FILLED,
                CommandState.FILLED,
                CommandState.CANCELLED,
                CommandState.CANCEL_REQUESTED,
            }:
                raise WorkerDegraded("native_lifecycle_command_not_dispatched")
            deadline = min(
                job.deadline, job.lease_expires_at or started, started + timedelta(seconds=90)
            )

            def validate():
                checked = self.clock()
                if (
                    checked < started
                    or checked >= deadline
                    or self.jobs.get(job.id) != job
                    or job.status is not WorkerJobStatus.RUNNING
                    or self.store.account_scope(scope.id) != scope
                ):
                    raise WorkerDegraded("native_lifecycle_authority_changed")

            validate()
            worker = "native-" + sha256((job.id + ":" + str(job.fence)).encode()).hexdigest()[:24]
            claim = command.claim
            if claim is None or claim.worker_id != worker or claim.lease_expires_at <= self.clock():
                claim = self.store.claim_command(
                    command.id,
                    worker_id=worker,
                    now=self.clock(),
                    lease_seconds=max(1, int((deadline - self.clock()).total_seconds())),
                    validate=validate,
                )
            if claim is None:
                raise WorkerDegraded("native_lifecycle_command_busy")
            command = self.store.command_for_intent(intent)
            reader = self.resolve_reader(scope)
            if not isinstance(reader, NativeOrderReader) or reader.scope != scope:
                raise WorkerDegraded("native_lifecycle_reader_binding_changed")
            known = self.lifecycle.load(scope.id, command.id)
            followup = None
            followup_time = self.clock()
            followup_terminal = False

            def stage_followup(_projection):
                nonlocal followup
                if followup_terminal or recovery_round >= 16:
                    return
                due = max(
                    followup_time + timedelta(seconds=30),
                    claim.lease_expires_at + timedelta(microseconds=1),
                )
                if due >= job.deadline:
                    return
                payload = {
                    "command_id": command.id,
                    "account_epoch_id": str(scope.account_epoch),
                    "recovery_root_id": root_id,
                    "recovery_round_id": str(recovery_round + 1),
                }
                child_id = (
                    "native-followup-"
                    + sha256(
                        json.dumps(
                            [root_id, command.id, scope.account_epoch, recovery_round + 1],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest()
                )
                existing = self.jobs._client.get(self.jobs._key(child_id))
                if existing is not None:
                    child = self.jobs._from_entity(existing)
                    if (
                        child.account_scope_id != scope.id
                        or child.kind is not WorkerJobKind.RECOVERY
                        or child.payload != payload
                        or child.deadline != job.deadline
                        or child.not_before is None
                    ):
                        raise WorkerDegraded("native_lifecycle_followup_changed")
                else:
                    child = WorkerJob(
                        child_id,
                        scope.id,
                        WorkerJobKind.RECOVERY,
                        payload,
                        job.deadline,
                        created_at=followup_time,
                        not_before=due,
                    )
                followup = self.jobs._add_in_transaction(child)

            def validate_capture():
                validate()
                if (
                    self.store.command_for_intent(intent).claim != claim
                    or claim.lease_expires_at <= self.clock()
                ):
                    raise WorkerDegraded("native_lifecycle_command_claim_changed")

            history = NativeHistoryCapture(
                self.store._client,
                scope,
                command,
                guard=validate_capture,
                clock=self.clock,
                deadline=deadline,
            )
            try:
                observation = reader.capture(
                    command,
                    intent,
                    clock=self.clock,
                    guard=validate_capture,
                    known_order_id=known.venue_order_id if known else None,
                    history=history,
                )
            except NativeHistoryPending:
                followup_time = self.clock()
                with self.store._client.transaction():
                    validate_capture()
                    stage_followup(None)
                    validate_capture()
                self._dispatch_followup(followup)
                operations.add(1, {"venue": scope.venue, "outcome": "history_pending"})
                trace.get_current_span().set_attribute("outcome", "history_pending")
                return {
                    "status": "pending" if followup is not None else "reconciliation_required",
                    "reason": "native_history_incomplete",
                    "command_state": command.state.value,
                    "filled_quantity": str(known.filled_quantity) if known else "0",
                    "followup_job_id": followup.id if followup is not None else None,
                    "followup_budget_exhausted": followup is None,
                }
            followup_time = observation.observed_at
            followup_terminal = observation.terminal_evidence_complete
            validate()
            if (
                not isinstance(observation, NativeOrderObservation)
                or observation.scope_id != scope.id
                or observation.account_epoch != scope.account_epoch
                or observation.command_id != command.id
                or observation.intent_hash != intent.intent_hash
            ):
                raise WorkerDegraded("native_lifecycle_observation_binding_changed")
            if command.state in {CommandState.SUBMITTING, CommandState.SUBMISSION_UNKNOWN}:
                positive = VenueOrderLookup(
                    scope.id,
                    intent.instrument_id,
                    command.client_order_id,
                    command.request_fingerprint,
                    True,
                    True,
                    observation.observed_at,
                    scope.account_epoch,
                )
                recover_submission(
                    self.store,
                    command=command,
                    intent=intent,
                    claim=claim,
                    now=self.clock(),
                    lookups=[positive],
                    clock=self.clock,
                    validate=validate,
                )
                command = self.store.command_for_intent(intent)

            def stage_and_clear(projection):
                stage_followup(projection)
                history.clear_in_transaction()

            projection = reconcile_lifecycle(
                self.store,
                self.lifecycle,
                command=command,
                intent=intent,
                claim=claim,
                venue_order_id=observation.venue_order_id,
                fills=observation.fills,
                order_status=observation.order_status,
                observed_at=observation.observed_at,
                terminal_evidence_complete=observation.terminal_evidence_complete,
                clock=self.clock,
                validate=validate,
                stage_continuation=stage_and_clear,
            )
            current = self.store.command_for_intent(intent)
            operations.add(1, {"venue": scope.venue, "outcome": "committed"})
            trace.get_current_span().set_attribute("outcome", "committed")
            self._dispatch_followup(followup)
            return {
                "status": "complete"
                if observation.terminal_evidence_complete
                else ("pending" if followup is not None else "reconciliation_required"),
                "command_state": current.state.value,
                "filled_quantity": str(projection.filled_quantity),
                "pending_trades": observation.pending_trades,
                "followup_job_id": followup.id if followup is not None else None,
                "followup_budget_exhausted": not observation.terminal_evidence_complete
                and followup is None,
            }
        # aqg: top-level boundary -- raw account/provider and credential text stays private
        except Exception:
            if history is not None:
                try:
                    history.invalidate_complete_on_error()
                # aqg: top-level boundary -- stale workers must not change private checkpoints
                except Exception:
                    logger.warning("Native history refresh remains unconfirmed")
            operations.add(1, {"outcome": "degraded"})
            trace.get_current_span().set_status(
                Status(StatusCode.ERROR, "native_lifecycle_unconfirmed")
            )
            logger.warning("Native lifecycle requires reconciliation")
            raise WorkerDegraded("native_lifecycle_unconfirmed") from None
