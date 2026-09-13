"""Bounded, public-data capture for forward shadow strategy cycles."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol, Sequence

from opentelemetry import metrics, trace

from ..market_data import MarketDataError
from .market import normalize_market
from .models import Instrument, MarketSnapshot, SchemaValidationError

tracer = trace.get_tracer(__name__)
market_capture_attempts = metrics.get_meter(__name__).create_counter(
    "twin.market.capture_attempts", unit="1",
)


class MarketCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketCapturePolicy:
    max_candidates: int = 3
    candidates_per_venue: int = 3
    min_close_days: float = 1.0
    max_close_days: float = 30.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 6:
            raise MarketCaptureError("market capture candidate limit is invalid")
        if not 1 <= self.candidates_per_venue <= 6:
            raise MarketCaptureError("market discovery limit is invalid")
        if not 0 <= self.min_close_days < self.max_close_days <= 365:
            raise MarketCaptureError("market capture horizon is invalid")


@dataclass(frozen=True)
class CapturedMarket:
    instrument: Instrument
    snapshot: MarketSnapshot
    yes_ask_depth: Decimal
    no_ask_depth: Decimal
    settlement_rules: str

    def __post_init__(self) -> None:
        if self.snapshot.instrument_id != self.instrument.id:
            raise MarketCaptureError("captured market identity is inconsistent")
        for name in ("yes_ask_depth", "no_ask_depth"):
            try:
                value = Decimal(str(getattr(self, name)))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise MarketCaptureError("captured market depth is invalid") from exc
            if not value.is_finite() or value <= 0:
                raise MarketCaptureError("captured market depth must be positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.settlement_rules, str) or not self.settlement_rules.strip():
            raise MarketCaptureError("captured market needs exact settlement rules")

    def to_storage(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument.to_storage(),
            "snapshot": self.snapshot.to_storage(),
            "yes_ask_depth": str(self.yes_ask_depth),
            "no_ask_depth": str(self.no_ask_depth),
            "settlement_rules": self.settlement_rules,
        }

    @classmethod
    def from_storage(cls, payload: Any) -> "CapturedMarket":
        if not isinstance(payload, Mapping) or set(payload) != {
            "instrument", "snapshot", "yes_ask_depth", "no_ask_depth",
            "settlement_rules",
        }:
            raise MarketCaptureError("stored captured market schema is invalid")
        try:
            return cls(
                Instrument(**dict(payload["instrument"])),
                MarketSnapshot(**dict(payload["snapshot"])),
                Decimal(str(payload["yes_ask_depth"])),
                Decimal(str(payload["no_ask_depth"])),
                str(payload["settlement_rules"]),
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            raise MarketCaptureError("stored captured market is malformed") from exc


@dataclass(frozen=True)
class MarketCaptureRejection:
    venue: str
    identifier: str
    reason: str

    def __post_init__(self) -> None:
        if self.venue not in {"kalshi", "polymarket"}:
            raise MarketCaptureError("capture rejection venue is invalid")
        if not self.identifier.strip() or not self.reason.strip():
            raise MarketCaptureError("capture rejection is incomplete")


@dataclass(frozen=True)
class MarketCaptureBatch:
    observed_at: datetime
    markets: tuple[CapturedMarket, ...]
    rejections: tuple[MarketCaptureRejection, ...]

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise MarketCaptureError("market capture time must be timezone-aware")
        if len({item.instrument.id for item in self.markets}) != len(self.markets):
            raise MarketCaptureError("market capture contains duplicate instruments")

    def to_storage(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "observed_at": self.observed_at.isoformat(),
            "markets": [item.to_storage() for item in self.markets],
            "rejections": [
                {"venue": item.venue, "identifier": item.identifier, "reason": item.reason}
                for item in self.rejections
            ],
        }

    @classmethod
    def from_storage(cls, payload: Any) -> "MarketCaptureBatch":
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version", "observed_at", "markets", "rejections",
        } or payload.get("schema_version") != 1:
            raise MarketCaptureError("stored market capture schema is invalid")
        if not isinstance(payload["markets"], list) or not isinstance(payload["rejections"], list):
            raise MarketCaptureError("stored market capture collections are invalid")
        try:
            rejections = tuple(
                MarketCaptureRejection(
                    str(item["venue"]), str(item["identifier"]), str(item["reason"]),
                )
                for item in payload["rejections"]
                if isinstance(item, Mapping) and set(item) == {"venue", "identifier", "reason"}
            )
            if len(rejections) != len(payload["rejections"]):
                raise MarketCaptureError("stored market rejections are malformed")
            return cls(
                datetime.fromisoformat(str(payload["observed_at"])),
                tuple(CapturedMarket.from_storage(item) for item in payload["markets"]),
                rejections,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketCaptureError("stored market capture is malformed") from exc


class MarketCaptureStore(Protocol):
    durable: bool

    def record(self, cycle_id: str, batch: MarketCaptureBatch) -> bool: ...
    def get(self, cycle_id: str) -> Optional[MarketCaptureBatch]: ...


def _encoded(batch: MarketCaptureBatch) -> str:
    return json.dumps(
        batch.to_storage(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


class InMemoryMarketCaptureStore:
    durable = False

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, str] = {}

    def record(self, cycle_id: str, batch: MarketCaptureBatch) -> bool:
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise MarketCaptureError("market capture cycle ID is required")
        encoded = _encoded(batch)
        with self._lock:
            existing = self._items.get(cycle_id)
            if existing is not None:
                if existing != encoded:
                    raise MarketCaptureError("market capture cycle has conflicting observations")
                return False
            self._items[cycle_id] = encoded
        return True

    def get(self, cycle_id: str) -> Optional[MarketCaptureBatch]:
        with self._lock:
            encoded = self._items.get(str(cycle_id))
        return None if encoded is None else MarketCaptureBatch.from_storage(json.loads(encoded))


class DatastoreMarketCaptureStore:
    durable = True

    def __init__(self, client: Any, *, namespace: str = "foresea-twin") -> None:
        self._client = client
        self._namespace = namespace

    def _key(self, cycle_id: str):
        return self._client.key(
            "TwinMarketCapture", sha256(cycle_id.encode("utf-8")).hexdigest(),
            namespace=self._namespace,
        )

    @staticmethod
    def _restore(entity: Any) -> MarketCaptureBatch:
        encoded = str(entity.get("payload_json") or "")
        fingerprint = sha256(encoded.encode("utf-8")).hexdigest()
        if entity.get("fingerprint") != fingerprint:
            raise MarketCaptureError("stored market capture failed integrity validation")
        return MarketCaptureBatch.from_storage(json.loads(encoded))

    @tracer.start_as_current_span("twin.market.capture_store.record")
    def record(self, cycle_id: str, batch: MarketCaptureBatch) -> bool:
        from google.cloud import datastore

        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise MarketCaptureError("market capture cycle ID is required")
        encoded = _encoded(batch)
        if len(encoded.encode("utf-8")) > 900_000:
            raise MarketCaptureError("market capture exceeds the durable size limit")
        key = self._key(cycle_id)
        with self._client.transaction():
            existing = self._client.get(key)
            if existing is not None:
                if str(existing.get("cycle_id")) != cycle_id or str(existing.get("payload_json")) != encoded:
                    raise MarketCaptureError("market capture cycle has conflicting observations")
                self._restore(existing)
                return False
            entity = datastore.Entity(key=key, exclude_from_indexes=("payload_json",))
            entity.update({
                "cycle_id": cycle_id,
                "observed_at": batch.observed_at,
                "market_count": len(batch.markets),
                "rejection_count": len(batch.rejections),
                "payload_json": encoded,
                "fingerprint": sha256(encoded.encode("utf-8")).hexdigest(),
            })
            self._client.put(entity)
        return True

    def get(self, cycle_id: str) -> Optional[MarketCaptureBatch]:
        entity = self._client.get(self._key(str(cycle_id)))
        return None if entity is None else self._restore(entity)


class MarketDataGateway(Protocol):
    def discover(
        self, venue: str, *, limit: int, min_close_days: float,
        max_close_days: float,
    ) -> Sequence[Mapping[str, Any]]: ...

    def fetch(
        self, venue: str, identifier: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]: ...


class LiveMarketDataGateway:
    """Narrow adapter over Foresea's existing public venue clients."""

    def discover(
        self, venue: str, *, limit: int, min_close_days: float,
        max_close_days: float,
    ) -> Sequence[Mapping[str, Any]]:
        from ..market_data import list_kalshi, list_polymarket

        kwargs = {
            "limit": limit,
            "min_close_days": min_close_days,
            "max_close_days": max_close_days,
            "contested_only": True,
        }
        if venue == "kalshi":
            return list_kalshi(**kwargs)
        if venue == "polymarket":
            return list_polymarket(**kwargs)
        raise MarketCaptureError("market discovery venue is invalid")

    def fetch(
        self, venue: str, identifier: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
        from ..market_data import fetch_kalshi_orderbook, fetch_twin_market_payload

        market, books = fetch_twin_market_payload(venue, identifier)
        if venue == "kalshi":
            books = {identifier: fetch_kalshi_orderbook(identifier)}
        return market, books


def _decimal(raw: Any) -> Decimal | None:
    if raw in (None, ""):
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None


def _levels(raw: Any) -> list[tuple[Decimal, Decimal]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    levels: list[tuple[Decimal, Decimal]] = []
    for item in raw:
        if isinstance(item, Mapping):
            price = _decimal(item.get("price"))
            size = _decimal(item.get("size") or item.get("quantity"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            price, size = _decimal(item[0]), _decimal(item[1])
        else:
            continue
        if price is not None and size is not None and size > 0:
            levels.append((price, size))
    return levels


def _depth_at(levels: Sequence[tuple[Decimal, Decimal]], price: Decimal) -> Decimal:
    return sum((size for level_price, size in levels if level_price == price), Decimal("0"))


def _polymarket_depth(
    market: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]],
    snapshot: MarketSnapshot,
) -> tuple[Decimal, Decimal] | None:
    tokens = market.get("clobTokenIds")
    if isinstance(tokens, str):
        import json
        try:
            tokens = json.loads(tokens)
        except json.JSONDecodeError:
            return None
    if not isinstance(tokens, Sequence) or len(tokens) != 2:
        return None
    depths: list[Decimal] = []
    for token, ask in zip(tokens, (snapshot.yes_ask, snapshot.no_ask)):
        book = books.get(str(token))
        if not isinstance(book, Mapping) or ask is None:
            return None
        depths.append(_depth_at(_levels(book.get("asks")), ask))
    return (depths[0], depths[1]) if min(depths) > 0 else None


def _kalshi_depth(
    market: Mapping[str, Any], books: Mapping[str, Mapping[str, Any]],
    snapshot: MarketSnapshot,
) -> tuple[Decimal, Decimal] | None:
    direct = (
        _decimal(market.get("yes_ask_size_fp") or market.get("yes_ask_size")),
        _decimal(market.get("no_ask_size_fp") or market.get("no_ask_size")),
    )
    if direct[0] and direct[1]:
        return direct[0], direct[1]
    book = next(iter(books.values()), None)
    if not isinstance(book, Mapping):
        return None
    raw = book.get("orderbook_fp") or book.get("orderbook") or book
    if not isinstance(raw, Mapping):
        return None
    yes_bids = _levels(raw.get("yes_dollars") or raw.get("yes"))
    no_bids = _levels(raw.get("no_dollars") or raw.get("no"))
    if snapshot.yes_ask is None or snapshot.no_ask is None:
        return None
    yes_depth = _depth_at(no_bids, Decimal("1") - snapshot.yes_ask)
    no_depth = _depth_at(yes_bids, Decimal("1") - snapshot.no_ask)
    return (yes_depth, no_depth) if min(yes_depth, no_depth) > 0 else None


def _settlement_rules(venue: str, market: Mapping[str, Any]) -> str:
    if venue == "kalshi":
        raw = market.get("rules_primary") or market.get("settlement_value") or market.get("subtitle")
    else:
        raw = market.get("rules") or market.get("description") or market.get("resolutionSource")
    return str(raw or "").strip()


@tracer.start_as_current_span("twin.market.capture")
def capture_markets(
    gateway: MarketDataGateway, *, now: datetime,
    policy: MarketCapturePolicy | None = None, sequence_start: int = 1,
) -> MarketCaptureBatch:
    """Capture at most three deterministic executable market observations."""
    if now.tzinfo is None or now.utcoffset() is None or sequence_start < 1:
        raise MarketCaptureError("market capture clock or sequence is invalid")
    policy = policy or MarketCapturePolicy()
    discovered: dict[str, list[str]] = {"kalshi": [], "polymarket": []}
    rejections: list[MarketCaptureRejection] = []
    for venue in discovered:
        try:
            rows = gateway.discover(
                venue, limit=policy.candidates_per_venue,
                min_close_days=policy.min_close_days,
                max_close_days=policy.max_close_days,
            )
        except MarketDataError:
            rejections.append(MarketCaptureRejection(venue, "discovery", "data_unavailable"))
            continue
        for row in rows:
            identifier = str(row.get("market_id") or row.get("ident") or "").strip()
            if identifier and identifier not in discovered[venue]:
                discovered[venue].append(identifier)
    identities: list[tuple[str, str]] = []
    for index in range(policy.candidates_per_venue):
        for venue in ("kalshi", "polymarket"):
            if index < len(discovered[venue]):
                identities.append((venue, discovered[venue][index]))
    markets: list[CapturedMarket] = []
    for venue, identifier in identities:
        if len(markets) >= policy.max_candidates:
            break
        try:
            market, books = gateway.fetch(venue, identifier)
        except MarketDataError:
            rejections.append(MarketCaptureRejection(venue, identifier, "data_unavailable"))
            continue
        try:
            assessment = normalize_market(
                venue, market, received_at=now,
                sequence=sequence_start + len(markets), orderbooks=books,
                min_horizon_seconds=int(policy.min_close_days * 86400),
                max_horizon_seconds=int(policy.max_close_days * 86400),
            )
        except SchemaValidationError:
            rejections.append(MarketCaptureRejection(
                venue, identifier, "invalid_market_schema",
            ))
            continue
        if not assessment.eligible:
            reasons = ",".join(item.value for item in assessment.reasons)
            rejections.append(MarketCaptureRejection(venue, identifier, reasons))
            continue
        assert assessment.instrument is not None and assessment.snapshot is not None
        depth = (
            _kalshi_depth(market, books, assessment.snapshot)
            if venue == "kalshi"
            else _polymarket_depth(market, books, assessment.snapshot)
        )
        if depth is None:
            rejections.append(MarketCaptureRejection(venue, identifier, "missing_executable_depth"))
            continue
        markets.append(CapturedMarket(
            assessment.instrument, assessment.snapshot, depth[0], depth[1],
            _settlement_rules(venue, market),
        ))
        market_capture_attempts.add(1, {"venue": venue, "outcome": "captured"})
    return MarketCaptureBatch(now, tuple(markets), tuple(rejections))
