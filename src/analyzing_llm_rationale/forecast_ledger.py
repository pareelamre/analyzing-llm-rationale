"""Append-only audit ledger for point-in-time forecasts and resolutions."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Protocol

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from analyzing_llm_rationale.trackrec_store import Entity

LEDGER_KIND = "ForecastLedgerEvent"
FORECAST_RECORDED = "forecast_recorded"
MARKET_RESOLVED = "market_resolved"
SCHEMA_VERSION = 1
MAX_AUDIT_INGEST_DELAY_SECONDS = 6 * 60 * 60

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("foresea.forecast_ledger")
meter = metrics.get_meter("foresea.forecast_ledger")
ledger_syncs = meter.create_counter(
    "forecast_ledger.syncs", unit="1", description="Forecast ledger synchronization runs"
)
ledger_events = meter.create_counter(
    "forecast_ledger.events", unit="1", description="Immutable forecast ledger events appended"
)
ledger_sync_duration = meter.create_histogram(
    "forecast_ledger.sync.duration",
    unit="s",
    description="Forecast ledger synchronization latency",
)


class LedgerError(RuntimeError):
    """Base class for forecast ledger failures."""


class LedgerValidationError(LedgerError):
    """A forecast or resolution cannot be represented safely."""


class ImmutableEventConflict(LedgerError):
    """An existing immutable event disagrees with a newly supplied event."""


class LedgerStore(Protocol):
    def key(self, kind: str, id_: str): ...

    def get(self, key): ...

    def insert_immutable(self, entity: Entity) -> bool: ...

    def query(self, kind: str): ...


def _iso_utc(value: Any, *, field: str) -> str:
    if isinstance(value, dict):
        value = value.get("__dt__")
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise LedgerValidationError(f"{field} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise LedgerValidationError(f"{field} is required")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _probability(value: Any, *, field: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"{field} must be a probability") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise LedgerValidationError(f"{field} must be between 0 and 1")
    return probability


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _iso_utc(value, field="event timestamp")
    return str(value)


def _market_key(platform: Any, ident: Any) -> tuple[str, str]:
    normalized_platform = str(platform or "").strip().lower()
    normalized_ident = str(ident or "").strip()
    if not normalized_platform or not normalized_ident:
        raise LedgerValidationError("platform and ident are required")
    return normalized_platform, normalized_ident


class ForecastLedger:
    """Write immutable, idempotent events through an existing track-record store."""

    def __init__(self, store: LedgerStore):
        self.store = store
        self._events = {
            entity.key.id: entity
            for entity in self.store.query(kind=LEDGER_KIND).fetch()
        }

    def _append(
        self,
        *,
        event_id: str,
        event: Mapping[str, Any],
        ingest_state: str | None = None,
    ) -> bool:
        canonical = _canonical_json(event)
        payload_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        key = self.store.key(LEDGER_KIND, event_id)
        existing = self._events.get(event_id)
        if existing is not None:
            if existing.get("payload_sha256") != payload_sha256:
                raise ImmutableEventConflict(f"immutable ledger event conflicts: {event_id}")
            return False

        entity = Entity(key, exclude_from_indexes=("payload",))
        entity.update(
            event_type=event["event_type"],
            forecast_id=event.get("forecast_id"),
            platform=event["platform"],
            ident=event["ident"],
            model=event.get("model"),
            event_ts=event["event_ts"],
            ingested_at=_now_utc(),
            ingest_state=ingest_state,
            payload_sha256=payload_sha256,
            payload=dict(event),
        )
        inserted = self.store.insert_immutable(entity)
        if inserted:
            self._events[event_id] = entity
            return True

        existing = self.store.get(key)
        if existing is None or existing.get("payload_sha256") != payload_sha256:
            raise ImmutableEventConflict(f"immutable ledger event conflicts: {event_id}")
        return False

    def record_forecast(self, snapshot: Mapping[str, Any], *, snapshot_key: str) -> bool:
        platform, ident = _market_key(snapshot.get("platform"), snapshot.get("ident"))
        forecasted_at = _iso_utc(snapshot.get("snapshot_ts"), field="snapshot_ts")
        forecasted_dt = datetime.fromisoformat(forecasted_at)
        close_time = (
            _iso_utc(snapshot["close_time"], field="close_time")
            if snapshot.get("close_time") is not None
            else None
        )
        forecast_before_close = (
            close_time is not None
            and forecasted_dt < datetime.fromisoformat(close_time)
        )
        model_probability = _probability(
            snapshot.get("model_probability"), field="model_probability"
        )
        market_probability = _probability(
            snapshot.get("market_probability"), field="market_probability"
        )
        evidence_as_of = snapshot.get("evidence_as_of")
        if evidence_as_of is not None:
            evidence_as_of = _iso_utc(evidence_as_of, field="evidence_as_of")
            if evidence_as_of > forecasted_at:
                raise LedgerValidationError("evidence_as_of cannot be after snapshot_ts")

        identity = {
            "snapshot_key": str(snapshot_key),
            "forecasted_at": forecasted_at,
            "model_probability": model_probability,
            "market_probability": market_probability,
        }
        forecast_id = _digest(identity)
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": FORECAST_RECORDED,
            "event_ts": forecasted_at,
            "forecast_id": forecast_id,
            "snapshot_key": str(snapshot_key),
            "platform": platform,
            "ident": ident,
            "model": str(snapshot.get("model") or "unknown"),
            "model_version": str(
                snapshot.get("model_version") or snapshot.get("model") or "unknown"
            ),
            "question": str(snapshot.get("question") or ""),
            "forecasted_at": forecasted_at,
            "evidence_as_of": evidence_as_of,
            "close_time": close_time,
            "domain": str(snapshot.get("domain") or "other"),
            "horizon": str(snapshot.get("horizon") or "unknown"),
            "model_probability": model_probability,
            "market_probability": market_probability,
            "market_bid": _optional_float(snapshot.get("market_bid")),
            "market_ask": _optional_float(snapshot.get("market_ask")),
            "source": str(snapshot.get("source") or "track_record_snapshot"),
        }
        return self._append(
            event_id=f"forecast:{forecast_id}",
            event=event,
            ingest_state=(
                "resolved"
                if snapshot.get("resolved")
                else (
                    "post_close"
                    if close_time is not None and not forecast_before_close
                    else "open"
                )
            ),
        )

    def record_resolution(
        self,
        *,
        platform: Any,
        ident: Any,
        outcome: Any,
        resolved_at: Any,
        source: str = "market_resolution",
    ) -> bool:
        normalized_platform, normalized_ident = _market_key(platform, ident)
        if outcome not in {0, 1}:
            raise LedgerValidationError("outcome must be binary")
        resolution_identity = {
            "platform": normalized_platform,
            "ident": normalized_ident,
        }
        resolution_id = _digest(resolution_identity)
        event_id = f"resolution:{resolution_id}"
        existing = self._events.get(event_id)
        if existing is not None:
            payload = existing.get("payload") or {}
            if int(payload.get("outcome")) != int(outcome):
                raise ImmutableEventConflict(
                    f"market resolution conflicts: {normalized_platform}:{normalized_ident}"
                )
            return False

        event_ts = _iso_utc(resolved_at, field="resolved_at")
        event = {
            "schema_version": SCHEMA_VERSION,
            "event_type": MARKET_RESOLVED,
            "event_ts": event_ts,
            "platform": normalized_platform,
            "ident": normalized_ident,
            "outcome": int(outcome),
            "resolved_at": event_ts,
            "source": source,
        }
        return self._append(event_id=event_id, event=event)

    def resolved_forecasts(self) -> list[Dict[str, Any]]:
        resolutions: Dict[tuple[str, str], tuple[Mapping[str, Any], Entity]] = {}
        forecasts: list[tuple[Mapping[str, Any], Entity]] = []
        for entity in self._events.values():
            payload = entity.get("payload") or {}
            if entity.get("event_type") == MARKET_RESOLVED:
                resolutions[(entity.get("platform"), entity.get("ident"))] = (payload, entity)
            elif entity.get("event_type") == FORECAST_RECORDED:
                forecasts.append((payload, entity))

        resolved: list[Dict[str, Any]] = []
        for forecast, forecast_entity in forecasts:
            resolution_item = resolutions.get(
                (forecast.get("platform"), forecast.get("ident"))
            )
            if resolution_item is None:
                continue
            resolution, resolution_entity = resolution_item
            ingested_at = _event_timestamp(forecast_entity.get("ingested_at"))
            forecasted_at = _iso_utc(forecast["forecasted_at"], field="forecasted_at")
            close_time = forecast.get("close_time")
            forecast_before_close = False
            if close_time is not None:
                forecast_before_close = (
                    datetime.fromisoformat(forecasted_at)
                    < datetime.fromisoformat(
                        _iso_utc(close_time, field="close_time")
                    )
                )
            audit_delay_seconds = None
            audit_grade = False
            if ingested_at is not None:
                ingested_dt = datetime.fromisoformat(ingested_at)
                resolved_dt = datetime.fromisoformat(
                    _iso_utc(resolution["resolved_at"], field="resolved_at")
                )
                delay = (
                    ingested_dt - datetime.fromisoformat(forecasted_at)
                ).total_seconds()
                audit_delay_seconds = delay
                audit_grade = (
                    forecast_entity.get("ingest_state") == "open"
                    and forecast_before_close
                    and ingested_dt <= resolved_dt
                    and 0.0 <= delay <= MAX_AUDIT_INGEST_DELAY_SECONDS
                )
            resolved.append(
                {
                    **forecast,
                    "outcome": int(resolution["outcome"]),
                    "resolved_at": resolution["resolved_at"],
                    "resolution_source": resolution.get("source"),
                    "ledger_ingested_at": ingested_at,
                    "resolution_ledger_ingested_at": _event_timestamp(
                        resolution_entity.get("ingested_at")
                    ),
                    "ledger_forecast_before_close": forecast_before_close,
                    "ledger_audit_delay_seconds": audit_delay_seconds,
                    "ledger_audit_grade": audit_grade,
                }
            )
        return sorted(
            resolved,
            key=lambda row: (row["resolved_at"], row["forecasted_at"], row["forecast_id"]),
        )


def sync_snapshot_ledger(store: LedgerStore) -> Dict[str, int]:
    """Idempotently mirror current legacy snapshots into immutable ledger events."""
    started = time.perf_counter()
    with tracer.start_as_current_span("forecast_ledger.sync") as span:
        try:
            # Read-only: hydrate question/domain/close_time/etc. from `markets`
            # for rows written after those fields stopped being duplicated onto
            # every snapshot. Safe here because these objects are only ever
            # passed into record_forecast()/record_resolution(), never put()
            # back onto forecast_snapshot.
            snapshots = getattr(store, "hydrate_markets", lambda rows: rows)(
                list(store.query(kind="ForecastSnapshot").fetch())
            )
            ledger = ForecastLedger(store)
            forecast_events = 0
            resolution_events = 0
            transaction = getattr(store, "transaction", None)

            with transaction() if transaction is not None else nullcontext():
                for snapshot in sorted(
                    snapshots,
                    key=lambda row: (
                        str(row.get("snapshot_ts") or ""),
                        str(getattr(row, "key", None)),
                    ),
                ):
                    if ledger.record_forecast(snapshot, snapshot_key=str(snapshot.key.id)):
                        forecast_events += 1

                resolved_snapshots = sorted(
                    (snapshot for snapshot in snapshots if snapshot.get("resolved")),
                    key=lambda row: str(row.get("resolved_ts") or ""),
                )
                for snapshot in resolved_snapshots:
                    if ledger.record_resolution(
                        platform=snapshot.get("platform"),
                        ident=snapshot.get("ident"),
                        outcome=snapshot.get("outcome"),
                        resolved_at=snapshot.get("resolved_ts"),
                        source="track_record_resolution",
                    ):
                        resolution_events += 1

            total = forecast_events + resolution_events
            span.set_attributes(
                {
                    "outcome": "success",
                    "snapshots.count": len(snapshots),
                    "events.count": total,
                }
            )
            ledger_syncs.add(1, {"outcome": "success"})
            if forecast_events:
                ledger_events.add(
                    forecast_events,
                    {"event_type": FORECAST_RECORDED, "outcome": "appended"},
                )
            if resolution_events:
                ledger_events.add(
                    resolution_events,
                    {"event_type": MARKET_RESOLVED, "outcome": "appended"},
                )
            result = {
                "snapshots_scanned": len(snapshots),
                "forecast_events_appended": forecast_events,
                "resolution_events_appended": resolution_events,
            }
            logger.info("forecast ledger sync completed: %s", result)
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("outcome", "failure")
            ledger_syncs.add(1, {"outcome": "failure"})
            logger.exception("forecast ledger sync failed")
            raise
        finally:
            ledger_sync_duration.record(
                time.perf_counter() - started,
                {"operation": "snapshot_sync"},
            )
