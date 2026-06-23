"""Store backends for the live track record.

Two implementations with the same API (both mimic the minimal
``google.cloud.datastore`` surface used by ``track_record_live.py``):

- ``FileStore``   — original JSON file (kept for migration / fallback)
- ``DuckDBStore`` — DuckDB single-file database (default for new ticks)

``DuckDBStore`` stores entities in two typed tables
(``forecast_snapshot``, ``market_price_point``); the public aggregate
(``static/track_record_live.json``) is written by the tick as before and
served by Cloud Run unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ── datetime-aware JSON (FileStore only) ─────────────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        dt = o if o.tzinfo else o.replace(tzinfo=timezone.utc)
        return {"__dt__": dt.isoformat()}
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


def _json_object_hook(d: Dict[str, Any]) -> Any:
    if len(d) == 1 and "__dt__" in d:
        try:
            return datetime.fromisoformat(d["__dt__"])
        except (TypeError, ValueError):
            return d
    return d


# ── Datastore-shaped primitives (shared) ─────────────────────────────────────

class Entity(dict):
    """A ``dict`` carrying a ``.key`` — drop-in for ``datastore.Entity``."""

    def __init__(self, key: "Key" = None, exclude_from_indexes: Tuple[str, ...] = ()):
        super().__init__()
        self.key = key
        self.exclude_from_indexes = exclude_from_indexes


class Key:
    def __init__(self, kind: str, id_: Optional[str] = None):
        self.kind = kind
        self.id = id_
        self.name = id_

    def __repr__(self) -> str:
        return f"Key({self.kind!r}, {self.id!r})"


class _Query:
    """In-memory query for FileStore."""

    def __init__(self, store: "FileStore", kind: str):
        self._store = store
        self._kind = kind
        self._filters: List[Tuple[str, str, Any]] = []

    def add_filter(self, name: str, op: str, value: Any) -> "_Query":
        self._filters.append((name, op, value))
        return self

    def fetch(self) -> Iterator[Entity]:
        for (kind, _id), entity in self._store.items():
            if kind != self._kind:
                continue
            if all(entity.get(n) == v for n, _op, v in self._filters):
                yield entity


# ── FileStore ─────────────────────────────────────────────────────────────────

class FileStore:
    """Persistent, single-file JSON store. Kept for migration and fallback."""

    def __init__(self, path: str | Path, *, autosave: bool = True):
        self.path = Path(path)
        self.autosave = autosave
        self._data: Dict[Tuple[str, str], Entity] = {}
        self.load()

    def load(self) -> "FileStore":
        self._data = {}
        if not self.path.exists():
            return self
        raw = json.loads(self.path.read_text(), object_hook=_json_object_hook)
        for kind, by_id in raw.items():
            for id_, fields in by_id.items():
                ent = Entity(Key(kind, id_))
                ent.update(fields)
                self._data[(kind, id_)] = ent
        return self

    def save(self) -> None:
        nested: Dict[str, Dict[str, Any]] = {}
        for (kind, id_), entity in self._data.items():
            nested.setdefault(kind, {})[id_] = dict(entity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(nested, default=_json_default,
                                  indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def key(self, kind: str, id_: Optional[str] = None) -> Key:
        return Key(kind, id_)

    def get(self, key: Key) -> Optional[Entity]:
        return self._data.get((key.kind, key.id))

    def put(self, entity: Entity) -> None:
        self._data[(entity.key.kind, entity.key.id)] = entity
        if self.autosave:
            self.save()

    def query(self, kind: str) -> _Query:
        return _Query(self, kind)

    def items(self) -> Iterator[Tuple[Tuple[str, str], Entity]]:
        return iter(list(self._data.items()))

    def count(self, kind: str) -> int:
        return sum(1 for (k, _i) in self._data if k == kind)


# ── DuckDBStore ───────────────────────────────────────────────────────────────

_SNAPSHOT_TABLE = "forecast_snapshot"
_PRICE_TABLE = "market_price_point"

# Columns and their DuckDB types for each table.
_SNAPSHOT_COLS: Dict[str, str] = {
    "key": "TEXT",
    "platform": "TEXT", "ident": "TEXT", "model": "TEXT",
    "snapshot_date": "TEXT", "snapshot_ts": "TEXT",
    "question": "TEXT", "market_url": "TEXT",
    "description": "TEXT", "resolution_criteria": "TEXT",
    "publish_time": "TEXT",
    "model_probability": "DOUBLE", "market_probability": "DOUBLE",
    "market_bid": "DOUBLE", "market_ask": "DOUBLE",
    "close_time": "TEXT", "lead_time_days": "DOUBLE", "horizon": "TEXT",
    "category": "TEXT", "domain": "TEXT",
    "market_volume": "DOUBLE", "market_liquidity": "DOUBLE",
    "evidence_count": "INTEGER", "entities": "TEXT",
    "resolved": "BOOLEAN", "outcome": "INTEGER", "resolved_ts": "TEXT",
    "model_brier": "DOUBLE", "market_brier": "DOUBLE", "model_correct": "BOOLEAN",
    "rationale": "TEXT",
}

_PRICE_COLS: Dict[str, str] = {
    "key": "TEXT",
    "platform": "TEXT", "ident": "TEXT", "market_url": "TEXT",
    "ts": "TEXT", "hour": "TEXT", "market_probability": "DOUBLE",
    "market_bid": "DOUBLE", "market_ask": "DOUBLE",
    "market_volume": "DOUBLE", "market_liquidity": "DOUBLE",
    "last_trade_price": "DOUBLE",
}

_KIND_TABLE = {
    "ForecastSnapshot": (_SNAPSHOT_TABLE, _SNAPSHOT_COLS),
    "MarketPricePoint": (_PRICE_TABLE, _PRICE_COLS),
}

# Fields that hold datetime objects — serialised as ISO strings in DuckDB.
_DT_FIELDS = {"snapshot_ts", "close_time", "resolved_ts", "ts"}


def _to_db(v: Any, col: str) -> Any:
    """Coerce a Python value to something DuckDB will accept."""
    if v is None:
        return None
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(v, dict) and "__dt__" in v:
        return v["__dt__"]
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v


def _from_db(v: Any, col: str) -> Any:
    """Coerce a DuckDB value back to the Python type track_record_live expects."""
    if v is None:
        return None
    if col in _DT_FIELDS and isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return v
    if col == "entities" and isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    return v


class _DuckQuery:
    def __init__(self, store: "DuckDBStore", kind: str):
        self._store = store
        self._kind = kind
        self._filters: List[Tuple[str, str, Any]] = []

    def add_filter(self, name: str, op: str, value: Any) -> "_DuckQuery":
        self._filters.append((name, op, value))
        return self

    def fetch(self) -> Iterator[Entity]:
        info = _KIND_TABLE.get(self._kind)
        if info is None:
            return
        table, cols = info
        where_parts, params = [], []
        for name, op, value in self._filters:
            if op == "=":
                where_parts.append(f"{name} = ?")
                params.append(_to_db(value, name))
        where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        rows = self._store._con.execute(
            f"SELECT * FROM {table} {where}", params
        ).fetchall()
        col_names = [desc[0] for desc in self._store._con.description]
        for row in rows:
            yield self._store._row_to_entity(self._kind, col_names, row)


class DuckDBStore:
    """DuckDB-backed store — same API as FileStore, SQL under the hood.

    The database is a single ``.duckdb`` file committed to git alongside the
    JSON aggregate. Only the GitHub Actions tick writes it; Cloud Run reads
    the JSON aggregate via GitHub raw URL as before.
    """

    def __init__(self, path: str | Path):
        import duckdb  # type: ignore[import]
        self.path = Path(path)
        self._con = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        for table, cols in ((_SNAPSHOT_TABLE, _SNAPSHOT_COLS),
                             (_PRICE_TABLE, _PRICE_COLS)):
            col_defs = ", ".join(f"{c} {t}" for c, t in cols.items())
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ({col_defs}, PRIMARY KEY (key))"
            )
            # Add any columns that exist in the schema but not in the table (migration).
            existing = {
                row[0] for row in
                self._con.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col, dtype in cols.items():
                if col not in existing:
                    self._con.execute(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {dtype}"
                    )

    # -- Datastore-compatible surface --

    def key(self, kind: str, id_: Optional[str] = None) -> Key:
        return Key(kind, id_)

    def get(self, key: Key) -> Optional[Entity]:
        info = _KIND_TABLE.get(key.kind)
        if info is None:
            return None
        table, cols = info
        rows = self._con.execute(
            f"SELECT * FROM {table} WHERE key = ?", [key.id]
        ).fetchall()
        if not rows:
            return None
        col_names = [desc[0] for desc in self._con.description]
        return self._row_to_entity(key.kind, col_names, rows[0])

    def put(self, entity: Entity) -> None:
        info = _KIND_TABLE.get(entity.key.kind)
        if info is None:
            return
        table, cols = info
        row: Dict[str, Any] = {"key": entity.key.id}
        for col in cols:
            if col == "key":
                continue
            row[col] = _to_db(entity.get(col), col)
        col_str = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        self._con.execute(
            f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
            list(row.values()),
        )

    def query(self, kind: str) -> _DuckQuery:
        return _DuckQuery(self, kind)

    def save(self) -> None:
        self._con.commit()

    def load(self) -> "DuckDBStore":
        return self

    def items(self) -> Iterator[Tuple[Tuple[str, str], Entity]]:
        for kind, (table, _cols) in _KIND_TABLE.items():
            rows = self._con.execute(f"SELECT * FROM {table}").fetchall()
            col_names = [desc[0] for desc in self._con.description]
            for row in rows:
                entity = self._row_to_entity(kind, col_names, row)
                yield (kind, entity.key.id), entity

    def count(self, kind: str) -> int:
        info = _KIND_TABLE.get(kind)
        if info is None:
            return 0
        table, _ = info
        return self._con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # -- helpers --

    def _row_to_entity(self, kind: str, col_names: List[str], row: tuple) -> Entity:
        key_val = None
        ent = Entity(None)
        for col, val in zip(col_names, row):
            if col == "key":
                key_val = val
            else:
                ent[col] = _from_db(val, col)
        ent.key = Key(kind, key_val)
        return ent

    def close(self) -> None:
        self._con.close()
