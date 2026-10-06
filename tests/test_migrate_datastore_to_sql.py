"""The Datastore -> SQL copier must preserve keys, namespaces, and values.

The migration runs once, against live production data, and a silent key or
namespace mistake would strand user accounts on the old backend. These tests
drive ``migrate`` and ``verify`` with a stub source client so the copy logic is
exercised without GCP credentials.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import datastore_sql  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_datastore_to_sql.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_datastore_to_sql", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubSource:
    """A Datastore-shaped source backed by the shim, plus kind/namespace metadata.

    Entities are re-wrapped with ``_DatastoreKey`` so the copier sees the same
    key shape the real client produces.
    """

    def __init__(self, inner: datastore_sql.Client, kinds, namespaces):
        self._inner = inner
        self._kinds = kinds
        self._namespaces = namespaces

    def query(self, kind, namespace=None):
        if kind == "__kind__":
            return _MetaQuery([_MetaKey(name) for name in self._kinds])
        if kind == "__namespace__":
            return _MetaQuery([_MetaKey(name) for name in self._namespaces])
        return _EntityQuery([
            _to_datastore_shaped(entity)
            for entity in self._inner.query(kind, namespace=namespace).fetch()
        ])


def _to_datastore_shaped(entity):
    """Convert a shim entity into one carrying a real-Datastore-shaped key."""
    path = [{"kind": k, "name": i} for k, i in entity.key.path]
    shaped = _DatastoreEntity(_DatastoreKey(path))
    shaped.update(dict(entity))
    return shaped


class _MetaKey:
    def __init__(self, name):
        self.name = name


class _MetaQuery:
    def __init__(self, keys):
        self._keys = keys

    def keys_only(self):
        return self

    def fetch(self):
        return [_MetaEntity(key) for key in self._keys]


class _MetaEntity:
    def __init__(self, key):
        self.key = key


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.source = datastore_sql.Client(path=str(Path(self._dir.name) / "source.sqlite3"))
        self.addCleanup(self.source.close)
        self.output = str(Path(self._dir.name) / "target.sqlite3")
        self.script = _load_script()

    def _seed(self):
        when = datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)
        user = datastore_sql.Entity(self.source.key("User", "u1"))
        user.update({"email": "u1@example.com", "created": when})
        self.source.put(user)

        convo = datastore_sql.Entity(self.source.key("User", "u1", "Conversation", "c1"))
        convo["title"] = "hello"
        self.source.put(convo)

        order = datastore_sql.Entity(self.source.key("User", "u1", "TradingOrder", "o1"))
        order.update({"size": Decimal("1.5"), "status": "running"})
        self.source.put(order)

        ns_entity = datastore_sql.Entity(
            self.source.key("TwinStrategyCycle", "s1", namespace="twin")
        )
        ns_entity["payload_json"] = "{}"
        self.source.put(ns_entity)

    def _stub(self):
        return _StubSource(
            self.source,
            kinds=["User", "Conversation", "TradingOrder", "TwinStrategyCycle"],
            namespaces=["", "twin"],
        )

    def test_migrate_copies_every_entity(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        total = self.script.migrate("proj", self.output)
        self.assertEqual(total, 4)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        self.assertEqual(target.get(target.key("User", "u1"))["email"], "u1@example.com")
        self.assertEqual(target.get(target.key("User", "u1", "Conversation", "c1"))["title"], "hello")

    def test_migrate_preserves_rich_types(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        order = target.get(target.key("User", "u1", "TradingOrder", "o1"))
        self.assertEqual(order["size"], Decimal("1.5"))
        user = target.get(target.key("User", "u1"))
        self.assertEqual(user["created"], datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc))

    def test_migrate_preserves_ancestor_relationships(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        query = target.query(kind="Conversation", ancestor=target.key("User", "u1"))
        self.assertEqual([e["title"] for e in query.fetch()], ["hello"])

    def test_migrate_preserves_namespaces(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        self.assertEqual(
            len(list(target.query(kind="TwinStrategyCycle", namespace="twin").fetch())), 1
        )
        self.assertEqual(
            len(list(target.query(kind="TwinStrategyCycle", namespace="").fetch())), 0
        )

    def test_verify_passes_on_a_complete_copy(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        self.script.migrate("proj", self.output)
        self.assertEqual(self.script.verify("proj", self.output), 0)

    def test_verify_fails_when_an_entity_is_missing(self):
        self._seed()
        self.script._gcp_client = lambda project: self._stub()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        target.delete(target.key("User", "u1", "Conversation", "c1"))
        target.close()

        self.assertEqual(self.script.verify("proj", self.output), 1)


class _DatastoreKey:
    """Mimics ``google.cloud.datastore.key.Key`` exactly.

    The real ``key.path`` is a list of ``{"kind", "name"|"id"}`` dicts and
    ``flat_path`` is a tuple -- not a string, and not a list of pairs. An
    earlier version of the copier iterated the dicts' *keys* and silently
    wrote every entity under the literal path ``kind:name``. This stub exists
    so that class of bug cannot come back.
    """

    def __init__(self, path):
        self.path = path

    @property
    def flat_path(self):
        return tuple(
            element.get("name") if element.get("name") is not None else element.get("id")
            for element in self.path
        )


class _DatastoreEntity(dict):
    def __init__(self, key, **fields):
        super().__init__(**fields)
        self.key = key


class _DatastoreShapedSource:
    """A source whose keys look like the real Datastore client's."""

    def __init__(self, entities, kinds, namespaces=("",)):
        self._entities = entities
        self._kinds = kinds
        self._namespaces = namespaces

    def query(self, kind, namespace=None):
        if kind == "__kind__":
            return _MetaQuery([_MetaKey(name) for name in self._kinds])
        if kind == "__namespace__":
            return _MetaQuery([_MetaKey(name) for name in self._namespaces])
        return _EntityQuery([e for e in self._entities if e.key.path[-1]["kind"] == kind])


class _EntityQuery:
    def __init__(self, entities):
        self._entities = entities

    def fetch(self):
        return list(self._entities)


class RealDatastoreKeyShapeTests(unittest.TestCase):
    """The copier must read the real Datastore key API, not a guess at it."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.output = str(Path(self._dir.name) / "target.sqlite3")
        self.script = _load_script()

    def _source(self):
        return _DatastoreShapedSource(
            entities=[
                _DatastoreEntity(
                    _DatastoreKey([{"kind": "User", "name": "107207861254381644111"}]),
                    email="u@example.com",
                ),
                _DatastoreEntity(
                    _DatastoreKey([
                        {"kind": "User", "name": "107207861254381644111"},
                        {"kind": "Conversation", "name": "conv-1"},
                    ]),
                    title="hello",
                ),
                _DatastoreEntity(
                    _DatastoreKey([{"kind": "AnalyticsEvent", "id": 5639445604728832}]),
                    event_name="forecast_completed",
                ),
            ],
            kinds=["User", "Conversation", "AnalyticsEvent"],
        )

    def test_named_keys_keep_their_identifier(self):
        self.script._gcp_client = lambda project: self._source()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        entity = target.get(target.key("User", "107207861254381644111"))
        self.assertIsNotNone(entity, "named key was not stored under its real id")
        self.assertEqual(entity["email"], "u@example.com")

    def test_numeric_ids_are_preserved(self):
        self.script._gcp_client = lambda project: self._source()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        entity = target.get(target.key("AnalyticsEvent", "5639445604728832"))
        self.assertIsNotNone(entity, "auto-allocated numeric id was lost")
        self.assertEqual(entity["event_name"], "forecast_completed")

    def test_ancestor_path_is_preserved(self):
        self.script._gcp_client = lambda project: self._source()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        query = target.query(
            kind="Conversation", ancestor=target.key("User", "107207861254381644111")
        )
        self.assertEqual([e["title"] for e in query.fetch()], ["hello"])

    def test_no_entity_lands_under_a_literal_kind_name_path(self):
        # The exact symptom of the original bug.
        self.script._gcp_client = lambda project: self._source()
        self.script.migrate("proj", self.output)

        target = datastore_sql.Client(path=self.output)
        self.addCleanup(target.close)
        self.assertIsNone(target.get(target.key("kind", "name")))
        self.assertIsNone(target.get(target.key("kind", "id")))

    def test_verify_passes_on_the_real_key_shape(self):
        self.script._gcp_client = lambda project: self._source()
        self.script.migrate("proj", self.output)
        self.assertEqual(self.script.verify("proj", self.output), 0)


if __name__ == "__main__":
    unittest.main()
