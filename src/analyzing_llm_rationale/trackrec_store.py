"""File-backed, Datastore-compatible store for the live track record.

The live track record used to live in Cloud Datastore, with the daily "tick"
(market fetch → LLM forecast → score → aggregate) running inside a single Cloud
Run request. That heavy request kept OOM/timeout-failing. We moved the whole loop
into a GitHub Action that commits the results back to the repo — so the data now
lives as a JSON file under version control (a git-scraping pattern), and Cloud Run
no longer runs any batch work.

To reuse ``track_record_live.py`` unchanged, this module mimics the *small* subset
of the ``google.cloud.datastore`` API that module relies on:

- ``Entity`` — a ``dict`` with a ``.key`` (so ``_ds.Entity(key, ...)`` keeps working)
- ``FileStore`` — the "client": ``key`` / ``get`` / ``put`` / ``query``

Entities are persisted to a single JSON file shaped ``{kind: {id: {fields}}}``.
``datetime`` values are round-tripped via a ``{"__dt__": "<iso>"}`` marker so the
trajectory timestamps survive serialization.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ── datetime-aware JSON ───────────────────────────────────────────────────────

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


# ── Datastore-shaped primitives ───────────────────────────────────────────────

class Entity(dict):
    """A ``dict`` carrying a ``.key`` — drop-in for ``datastore.Entity``."""

    def __init__(self, key: "Key" = None, exclude_from_indexes: Tuple[str, ...] = ()):
        super().__init__()
        self.key = key
        # Kept only for API-compatibility; a file store has no indexes.
        self.exclude_from_indexes = exclude_from_indexes


class Key:
    def __init__(self, kind: str, id_: Optional[str] = None):
        self.kind = kind
        self.id = id_
        self.name = id_  # datastore exposes string ids as ``.name``

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Key({self.kind!r}, {self.id!r})"


class _Query:
    def __init__(self, store: "FileStore", kind: str):
        self._store = store
        self._kind = kind
        self._filters: List[Tuple[str, str, Any]] = []

    def add_filter(self, name: str, op: str, value: Any) -> "_Query":
        # Only equality is used by track_record_live; that's all we support.
        self._filters.append((name, op, value))
        return self

    def fetch(self) -> Iterator[Entity]:
        for (kind, _id), entity in self._store.items():
            if kind != self._kind:
                continue
            if all(entity.get(n) == v for n, _op, v in self._filters):
                yield entity


# ── The "client" ──────────────────────────────────────────────────────────────

class FileStore:
    """Persistent, single-file, Datastore-compatible client.

    Not concurrency-safe — intended for a single writer (the GitHub Action). Reads
    in the server are done against a committed copy of the produced aggregate, not
    this store.
    """

    def __init__(self, path: str | Path, *, autosave: bool = True):
        self.path = Path(path)
        self.autosave = autosave
        self._data: Dict[Tuple[str, str], Entity] = {}
        self.load()

    # -- persistence --
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

    # -- datastore-compatible surface --
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

    # -- helpers --
    def items(self) -> Iterator[Tuple[Tuple[str, str], Entity]]:
        return iter(list(self._data.items()))

    def count(self, kind: str) -> int:
        return sum(1 for (k, _i) in self._data if k == kind)
