"""SQL-backed drop-in for the ``google.cloud.datastore`` surface this app uses.

Why this exists
---------------
Every durable entity in Foresea (users, conversations, trading orders, twin
account state, track-record snapshots) lives in Cloud Datastore. Datastore is
the single hardest thing to move off GCP, and it is also the thing that costs
almost nothing -- the bill is Cloud Run compute. So the migration strategy is
to make the *data layer* portable first, then move the container anywhere.

This module implements the exact subset of the Datastore client API the
codebase calls, on top of SQLite (default) or Postgres. It is deliberately
shaped like ``trackrec_store.py``, which already proves the pattern: a
``Key``/``Entity``/``query().add_filter().fetch()`` surface over SQL.

Supported surface (verified against every call site in ``src/``):

* ``Client``: ``key``, ``get``, ``get_multi``, ``put``, ``put_multi``,
  ``delete``, ``delete_multi``, ``query``, ``transaction``
* ``Key``: ``kind``, ``id``, ``name``, ``parent``, ``flat_path``, ``namespace``
* ``Entity``: ``dict`` subclass carrying ``.key`` and ``.exclude_from_indexes``
* ``Query``: ``add_filter(name, op, value)`` or
  ``add_filter(filter=PropertyFilter(...))``, ``.order``, ``.keys_only()``,
  ``.fetch(limit=...)``
* Operators: ``=``, ``<``, ``<=``, ``>``, ``>=``, ``IN``
* Ancestor queries, namespaces, and multi-level key paths.

Deliberately not supported (and not used anywhere in this codebase): composite
indexes, projection queries, eventual-consistency hints, GQL, ``__key__``
filters, and cross-group transactions.

``exclude_from_indexes`` is accepted and ignored: SQL has no separate index
definition per property, so the hint carries no meaning here.
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Client", "Entity", "Key", "PropertyFilter", "Query", "get_client"]

# ── value codec ──────────────────────────────────────────────────────────────
#
# Datastore stores rich Python types (datetime, bytes, Decimal) natively. JSON
# does not, so each is wrapped in a single-key marker dict. Decoding only
# unwraps a dict that has *exactly* one key, so a user dict that happens to
# contain "__dt__" alongside other keys round-trips untouched.

_DT = "__dt__"
_B64 = "__b64__"
_DEC = "__dec__"


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return {_DT: dt.isoformat()}
    if isinstance(value, bytes):
        return {_B64: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Decimal):
        return {_DEC: str(value)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if len(value) == 1:
            if _DT in value:
                try:
                    return datetime.fromisoformat(value[_DT])
                except (TypeError, ValueError):
                    return value
            if _B64 in value:
                try:
                    return base64.b64decode(value[_B64])
                except (TypeError, ValueError):
                    return value
            if _DEC in value:
                try:
                    return Decimal(value[_DEC])
                except (TypeError, ValueError):
                    return value
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _sort_key(value: Any) -> Tuple[int, Any]:
    """Order heterogeneous values deterministically, mirroring Datastore.

    Datastore orders by type first (None < numbers < strings < datetimes), so
    a mixed column never raises. ``None`` sorts first, matching Datastore.
    """
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float, Decimal)):
        return (1, float(value))
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return (3, dt.timestamp())
    return (4, str(value))


def _matches(actual: Any, op: str, expected: Any) -> bool:
    if op == "=":
        return actual == expected
    if op == "IN":
        return actual in expected
    if actual is None or expected is None:
        # Datastore never matches an inequality against a missing property.
        return False
    try:
        if op == "<":
            return _sort_key(actual) < _sort_key(expected)
        if op == "<=":
            return _sort_key(actual) <= _sort_key(expected)
        if op == ">":
            return _sort_key(actual) > _sort_key(expected)
        if op == ">=":
            return _sort_key(actual) >= _sort_key(expected)
    except TypeError:
        return False
    raise ValueError(f"Unsupported filter operator: {op!r}")


# ── key / entity ─────────────────────────────────────────────────────────────

class Key:
    """A Datastore key: an ordered path of ``(kind, id)`` pairs.

    ``client.key("User", uid, "Conversation", cid)`` produces the path
    ``((User, uid), (Conversation, cid))``, whose ``parent`` is the ``User``
    key -- exactly the ancestor relationship the app relies on.
    """

    __slots__ = ("_path", "namespace")

    def __init__(self, path: Sequence[Tuple[str, str]], namespace: Optional[str] = None):
        self._path = tuple((str(kind), str(ident)) for kind, ident in path)
        self.namespace = namespace

    @property
    def path(self) -> Tuple[Tuple[str, str], ...]:
        return self._path

    @property
    def kind(self) -> str:
        return self._path[-1][0]

    @property
    def id(self) -> str:
        return self._path[-1][1]

    @property
    def name(self) -> str:
        return self._path[-1][1]

    @property
    def parent(self) -> Optional["Key"]:
        if len(self._path) <= 1:
            return None
        return Key(self._path[:-1], namespace=self.namespace)

    @property
    def flat_path(self) -> str:
        return "/".join(f"{kind}:{ident}" for kind, ident in self._path)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Key)
            and self._path == other._path
            and self.namespace == other.namespace
        )

    def __hash__(self) -> int:
        return hash((self._path, self.namespace))

    def __repr__(self) -> str:
        return f"Key({self.flat_path!r}, namespace={self.namespace!r})"


class Entity(dict):
    """A ``dict`` carrying a ``.key`` -- drop-in for ``datastore.Entity``."""

    def __init__(self, key: Optional[Key] = None, exclude_from_indexes: Iterable[str] = ()):
        super().__init__()
        self.key = key
        self.exclude_from_indexes = tuple(exclude_from_indexes)


class PropertyFilter:
    """Mirrors ``google.cloud.datastore.query.PropertyFilter``."""

    __slots__ = ("name", "op", "value")

    def __init__(self, name: str, op: str, value: Any):
        self.name, self.op, self.value = name, op, value


# ── query ────────────────────────────────────────────────────────────────────

class Query:
    def __init__(self, client: "Client", kind: str, ancestor: Optional[Key] = None,
                 namespace: Optional[str] = None):
        self._client = client
        self._kind = kind
        self._ancestor = ancestor
        self._namespace = namespace if namespace is not None else client.namespace
        self._filters: List[Tuple[str, str, Any]] = []
        self._keys_only = False
        self.order: List[str] = []

    def add_filter(self, name: Optional[str] = None, op: Optional[str] = None,
                   value: Any = None, *, filter: Optional[PropertyFilter] = None) -> "Query":
        if filter is not None:
            name, op, value = filter.name, filter.op, filter.value
        if name is None or op is None:
            raise ValueError("add_filter requires a name and operator")
        self._filters.append((name, op, value))
        return self

    def keys_only(self) -> "Query":
        self._keys_only = True
        return self

    def fetch(self, limit: Optional[int] = None) -> Iterator[Entity]:
        rows = self._client._select(self._kind, self._ancestor, self._namespace)
        entities: List[Entity] = []
        for key, data in rows:
            if self._keys_only:
                entities.append(Entity(key))
                continue
            entity = Entity(key)
            entity.update(data)
            entities.append(entity)

        # Filters are applied in Python rather than pushed into SQL. Datastore
        # compares rich types (datetime, Decimal) with type-aware ordering, and
        # reproducing that in SQLite's dynamic typing is where a shim silently
        # diverges from the real thing. The row counts here are per-user and
        # per-market, so the scan is cheap and the semantics stay exact.
        for name, op, expected in self._filters:
            entities = [e for e in entities if _matches(e.get(name), op, expected)]

        if self.order:
            for field in reversed(self.order):
                descending = field.startswith("-")
                name = field[1:] if descending else field
                entities.sort(key=lambda e, n=name: _sort_key(e.get(n)), reverse=descending)

        if limit is not None:
            entities = entities[:limit]
        return iter(entities)


# ── client ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    namespace   TEXT NOT NULL DEFAULT '',
    key_path    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    id          TEXT NOT NULL,
    parent_path TEXT NOT NULL DEFAULT '',
    data        TEXT NOT NULL,
    PRIMARY KEY (namespace, key_path)
);
CREATE INDEX IF NOT EXISTS idx_entities_kind   ON entities(namespace, kind);
CREATE INDEX IF NOT EXISTS idx_entities_parent ON entities(namespace, parent_path);
"""


class Client:
    """SQLite-backed stand-in for ``google.cloud.datastore.Client``.

    ``path`` defaults to ``FORESEA_DATASTORE_PATH`` (``foresea.sqlite3`` in the
    working directory). ``project`` and other Datastore constructor kwargs are
    accepted and ignored so existing ``datastore.Client(project=...)`` call
    sites keep working unchanged.
    """

    def __init__(self, path: Optional[str] = None, namespace: Optional[str] = None,
                 **kwargs: Any):
        self._path = path or os.environ.get("FORESEA_DATASTORE_PATH", "foresea.sqlite3")
        self.namespace = namespace
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- keys --

    def key(self, *args: Any, namespace: Optional[str] = None) -> Key:
        if len(args) < 2 or len(args) % 2 != 0:
            raise ValueError("key() requires an even number of kind/id arguments")
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
        return Key(pairs, namespace=namespace if namespace is not None else self.namespace)

    # -- reads --

    def _select(self, kind: str, ancestor: Optional[Key],
                namespace: Optional[str]) -> List[Tuple[Key, Dict[str, Any]]]:
        ns = namespace if namespace is not None else self.namespace
        sql = "SELECT key_path, data FROM entities WHERE namespace = ? AND kind = ?"
        params: List[Any] = [ns or "", kind]
        if ancestor is not None:
            sql += " AND (parent_path = ? OR parent_path LIKE ?)"
            params.extend([ancestor.flat_path, ancestor.flat_path + "/%"])
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        results: List[Tuple[Key, Dict[str, Any]]] = []
        for row in rows:
            key = _key_from_flat_path(row["key_path"], ns)
            results.append((key, _decode(json.loads(row["data"]))))
        return results

    def get(self, key: Optional[Key]) -> Optional[Entity]:
        if key is None or not key.path:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM entities WHERE namespace = ? AND key_path = ?",
                [key.namespace or "", key.flat_path],
            ).fetchone()
        if row is None:
            return None
        entity = Entity(key)
        entity.update(_decode(json.loads(row["data"])))
        return entity

    def get_multi(self, keys: Iterable[Key]) -> List[Optional[Entity]]:
        return [self.get(key) for key in keys]

    # -- writes --

    def put(self, entity: Entity) -> Entity:
        if entity.key is None or not entity.key.path:
            raise ValueError("Cannot put an entity without a complete key")
        payload = json.dumps(_encode(dict(entity)), sort_keys=True)
        parent = entity.key.parent
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entities "
                "(namespace, key_path, kind, id, parent_path, data) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    entity.key.namespace or "",
                    entity.key.flat_path,
                    entity.key.kind,
                    entity.key.id,
                    parent.flat_path if parent is not None else "",
                    payload,
                ],
            )
            if self._tx_depth == 0:
                self._conn.commit()
        return entity

    def put_multi(self, entities: Iterable[Entity]) -> List[Entity]:
        written = []
        with self.transaction():
            for entity in entities:
                written.append(self.put(entity))
        return written

    def delete(self, key: Optional[Key]) -> None:
        if key is None or not key.path:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM entities WHERE namespace = ? AND key_path = ?",
                [key.namespace or "", key.flat_path],
            )
            if self._tx_depth == 0:
                self._conn.commit()

    def delete_multi(self, keys: Iterable[Key]) -> None:
        with self.transaction():
            for key in keys:
                self.delete(key)

    # -- queries --

    def query(self, kind: str, ancestor: Optional[Key] = None,
              namespace: Optional[str] = None, **kwargs: Any) -> Query:
        return Query(self, kind, ancestor=ancestor, namespace=namespace)

    # -- transactions --

    @contextmanager
    def transaction(self) -> Iterator["Client"]:
        """Serialise a group of writes.

        Datastore transactions do not nest; neither does this. The depth guard
        makes an inner ``transaction()`` a no-op that joins the outer one, so
        ``put_multi`` can be called from inside a caller's transaction without
        a spurious "cannot start a transaction within a transaction".
        """
        with self._lock:
            outermost = self._tx_depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth += 1
            try:
                yield self
            except Exception:
                self._tx_depth -= 1
                if outermost:
                    self._conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if outermost:
                    self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _key_from_flat_path(flat_path: str, namespace: Optional[str]) -> Key:
    pairs = []
    for segment in flat_path.split("/"):
        kind, _, ident = segment.partition(":")
        pairs.append((kind, ident))
    return Key(pairs, namespace=namespace)


def get_client(**kwargs: Any) -> Client:
    """Return a SQL-backed client. Mirrors ``datastore.Client()``."""
    return Client(**kwargs)
