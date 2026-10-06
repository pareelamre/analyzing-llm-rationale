"""The SQL shim must behave like the Datastore client it replaces.

These tests pin the semantics the app actually depends on -- ancestor
scoping, key paths, transactions, and the filter operators used across
``src/`` -- so a future change to ``datastore_sql`` cannot quietly diverge
from ``google.cloud.datastore`` and corrupt live trading state.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import datastore_sql as ds  # noqa: E402


class _ClientCase(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.client = ds.Client(path=str(Path(self._dir.name) / "test.sqlite3"))
        # Registered after the directory cleanup, so it runs first (LIFO) and
        # releases the file handle before Windows tries to delete the folder.
        self.addCleanup(self.client.close)

    def _put(self, kind: str, ident: str, **fields):
        key = self.client.key(kind, ident)
        entity = ds.Entity(key)
        entity.update(fields)
        self.client.put(entity)
        return key


class KeyTests(_ClientCase):
    def test_multi_level_path_and_parent(self):
        key = self.client.key("User", "u1", "Conversation", "c1")
        self.assertEqual(key.kind, "Conversation")
        self.assertEqual(key.name, "c1")
        self.assertEqual(key.id, "c1")
        self.assertEqual(key.parent.kind, "User")
        self.assertEqual(key.parent.name, "u1")
        self.assertIsNone(key.parent.parent)
        self.assertEqual(key.flat_path, "User:u1/Conversation:c1")

    def test_key_equality_and_hash(self):
        a = self.client.key("User", "u1")
        b = self.client.key("User", "u1")
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)

    def test_namespace_is_part_of_identity(self):
        a = self.client.key("TwinStrategyCycle", "x", namespace="ns1")
        b = self.client.key("TwinStrategyCycle", "x", namespace="ns2")
        self.assertNotEqual(a, b)

    def test_odd_argument_count_rejected(self):
        with self.assertRaises(ValueError):
            self.client.key("User")


class EntityRoundTripTests(_ClientCase):
    def test_rich_types_survive_a_round_trip(self):
        when = datetime(2026, 5, 29, 20, 52, 5, tzinfo=timezone.utc)
        key = self._put(
            "TwinAccount", "a1",
            when=when, amount=Decimal("12.34"), blob=b"\x00\xff",
            nested={"a": [1, {"b": when}]}, plain="text", number=7,
        )
        entity = self.client.get(key)
        self.assertEqual(entity["when"], when)
        self.assertEqual(entity["amount"], Decimal("12.34"))
        self.assertEqual(entity["blob"], b"\x00\xff")
        self.assertEqual(entity["nested"]["a"][1]["b"], when)
        self.assertEqual(entity["plain"], "text")
        self.assertEqual(entity["number"], 7)

    def test_naive_datetime_is_treated_as_utc(self):
        naive = datetime(2026, 1, 2, 3, 4, 5)
        key = self._put("AnalyticsEvent", "e1", ts=naive)
        self.assertEqual(self.client.get(key)["ts"], naive.replace(tzinfo=timezone.utc))

    def test_user_dict_with_marker_key_is_not_unwrapped(self):
        # A dict that merely *contains* "__dt__" alongside other keys is data,
        # not an encoded datetime, and must round-trip untouched.
        payload = {"__dt__": "not-a-date", "other": 1}
        key = self._put("Message", "m1", payload=payload)
        self.assertEqual(self.client.get(key)["payload"], payload)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.client.get(self.client.key("User", "absent")))

    def test_get_none_key_returns_none(self):
        self.assertIsNone(self.client.get(None))

    def test_put_without_key_rejected(self):
        with self.assertRaises(ValueError):
            self.client.put(ds.Entity())

    def test_put_is_upsert(self):
        key = self._put("User", "u1", email="a@example.com")
        entity = self.client.get(key)
        entity["email"] = "b@example.com"
        self.client.put(entity)
        self.assertEqual(self.client.get(key)["email"], "b@example.com")

    def test_exclude_from_indexes_is_accepted(self):
        key = self.client.key("TwinAccount", "a1")
        entity = ds.Entity(key, exclude_from_indexes=("payload_json",))
        entity["payload_json"] = "{}"
        self.client.put(entity)
        self.assertEqual(self.client.get(key)["payload_json"], "{}")


class AncestorQueryTests(_ClientCase):
    def setUp(self) -> None:
        super().setUp()
        self._put("User", "u1", email="u1@example.com")
        self._put("User", "u2", email="u2@example.com")
        for cid in ("c1", "c2"):
            key = self.client.key("User", "u1", "Conversation", cid)
            entity = ds.Entity(key)
            entity["title"] = cid
            self.client.put(entity)
        for mid in ("m1", "m2", "m3"):
            key = self.client.key("User", "u1", "Conversation", "c1", "Message", mid)
            entity = ds.Entity(key)
            entity["body"] = mid
            self.client.put(entity)

    def test_ancestor_query_is_scoped_to_that_parent(self):
        query = self.client.query(kind="Conversation", ancestor=self.client.key("User", "u1"))
        self.assertEqual({e["title"] for e in query.fetch()}, {"c1", "c2"})

    def test_ancestor_query_excludes_other_users(self):
        query = self.client.query(kind="Conversation", ancestor=self.client.key("User", "u2"))
        self.assertEqual(list(query.fetch()), [])

    def test_ancestor_query_includes_deeper_descendants(self):
        # Datastore ancestor queries match the whole subtree, not just direct
        # children -- the app relies on this for Message-under-Conversation.
        query = self.client.query(kind="Message", ancestor=self.client.key("User", "u1"))
        self.assertEqual({e["body"] for e in query.fetch()}, {"m1", "m2", "m3"})

    def test_kind_query_without_ancestor_spans_all_parents(self):
        self.assertEqual(len(list(self.client.query(kind="User").fetch())), 2)

    def test_entity_key_parent_is_recoverable(self):
        query = self.client.query(kind="Message", ancestor=self.client.key("User", "u1"))
        parents = {e.key.parent.name for e in query.fetch()}
        self.assertEqual(parents, {"c1"})


class FilterTests(_ClientCase):
    def setUp(self) -> None:
        super().setUp()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._put("AnalyticsEvent", "e1", ts=base, name="a")
        self._put("AnalyticsEvent", "e2", ts=base + timedelta(days=1), name="b")
        self._put("AnalyticsEvent", "e3", ts=base + timedelta(days=2), name="c")
        self._put("AnalyticsEvent", "e4", name="d")  # no ts

    def _names(self, query):
        return {e["name"] for e in query.fetch()}

    def test_equality(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("name", "=", "b")
        self.assertEqual(self._names(query), {"b"})

    def test_greater_than_or_equal_on_datetime(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("ts", ">=", datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertEqual(self._names(query), {"b", "c"})

    def test_less_than_on_datetime(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("ts", "<", datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertEqual(self._names(query), {"a"})

    def test_inequality_never_matches_missing_property(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("ts", ">=", datetime(2020, 1, 1, tzinfo=timezone.utc))
        self.assertNotIn("d", self._names(query))

    def test_in_operator(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("name", "IN", ["a", "c"])
        self.assertEqual(self._names(query), {"a", "c"})

    def test_property_filter_object_form(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter(filter=ds.PropertyFilter("name", "=", "c"))
        self.assertEqual(self._names(query), {"c"})

    def test_multiple_filters_are_anded(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("name", "IN", ["a", "b"])
        query.add_filter("ts", ">=", datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertEqual(self._names(query), {"b"})

    def test_order_ascending_and_descending(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.add_filter("name", "IN", ["a", "b", "c"])
        query.order = ["ts"]
        self.assertEqual([e["name"] for e in query.fetch()], ["a", "b", "c"])
        query.order = ["-ts"]
        self.assertEqual([e["name"] for e in query.fetch()], ["c", "b", "a"])

    def test_order_places_missing_values_first(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.order = ["ts"]
        self.assertEqual([e["name"] for e in query.fetch()][0], "d")

    def test_limit(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.order = ["name"]
        self.assertEqual([e["name"] for e in query.fetch(limit=2)], ["a", "b"])

    def test_keys_only_returns_keys_without_fields(self):
        query = self.client.query(kind="AnalyticsEvent")
        query.keys_only()
        entities = list(query.fetch())
        self.assertEqual(len(entities), 4)
        self.assertTrue(all(e.key is not None for e in entities))
        self.assertTrue(all(len(e) == 0 for e in entities))


class NamespaceTests(_ClientCase):
    def test_namespaces_are_isolated(self):
        for ns in ("ns1", "ns2"):
            key = self.client.key("TwinStrategyCycle", "same", namespace=ns)
            entity = ds.Entity(key)
            entity["ns"] = ns
            self.client.put(entity)
        q1 = self.client.query(kind="TwinStrategyCycle", namespace="ns1")
        self.assertEqual([e["ns"] for e in q1.fetch()], ["ns1"])


class TransactionTests(_ClientCase):
    def test_commit_persists_all_writes(self):
        with self.client.transaction():
            self._put("TwinAccount", "a1", cash="100")
            self._put("TwinAccount", "a2", cash="200")
        self.assertEqual(self.client.get(self.client.key("TwinAccount", "a1"))["cash"], "100")
        self.assertEqual(self.client.get(self.client.key("TwinAccount", "a2"))["cash"], "200")

    def test_rollback_discards_all_writes(self):
        with self.assertRaises(RuntimeError):
            with self.client.transaction():
                self._put("TwinAccount", "a1", cash="100")
                raise RuntimeError("boom")
        self.assertIsNone(self.client.get(self.client.key("TwinAccount", "a1")))

    def test_read_inside_transaction_sees_uncommitted_write(self):
        with self.client.transaction():
            self._put("TwinAccount", "a1", cash="100")
            self.assertEqual(self.client.get(self.client.key("TwinAccount", "a1"))["cash"], "100")

    def test_nested_transaction_joins_outer(self):
        # put_multi opens a transaction; calling it inside a caller's
        # transaction must not raise "cannot start a transaction within a
        # transaction".
        with self.client.transaction():
            self.client.put_multi([ds.Entity(self.client.key("TwinAccount", "a1"))])
            self.client.put_multi([ds.Entity(self.client.key("TwinAccount", "a2"))])
        self.assertIsNotNone(self.client.get(self.client.key("TwinAccount", "a1")))
        self.assertIsNotNone(self.client.get(self.client.key("TwinAccount", "a2")))

    def test_outer_rollback_also_discards_nested_writes(self):
        with self.assertRaises(RuntimeError):
            with self.client.transaction():
                self.client.put_multi([ds.Entity(self.client.key("TwinAccount", "a1"))])
                raise RuntimeError("boom")
        self.assertIsNone(self.client.get(self.client.key("TwinAccount", "a1")))


class MultiOperationTests(_ClientCase):
    def test_get_multi_preserves_order_and_gaps(self):
        self._put("User", "u1", email="1")
        self._put("User", "u3", email="3")
        keys = [self.client.key("User", i) for i in ("u1", "u2", "u3")]
        results = self.client.get_multi(keys)
        self.assertEqual([r["email"] if r else None for r in results], ["1", None, "3"])

    def test_delete_and_delete_multi(self):
        self._put("User", "u1")
        self._put("User", "u2")
        self.client.delete(self.client.key("User", "u1"))
        self.assertIsNone(self.client.get(self.client.key("User", "u1")))
        self.client.delete_multi([self.client.key("User", "u2")])
        self.assertIsNone(self.client.get(self.client.key("User", "u2")))

    def test_delete_missing_is_a_noop(self):
        self.client.delete(self.client.key("User", "absent"))


class PersistenceTests(unittest.TestCase):
    def test_data_survives_reopening_the_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "persist.sqlite3")
            first = ds.Client(path=path)
            entity = ds.Entity(first.key("User", "u1"))
            entity["email"] = "kept@example.com"
            first.put(entity)
            first.close()

            second = ds.Client(path=path)
            self.assertEqual(second.get(second.key("User", "u1"))["email"], "kept@example.com")
            second.close()


if __name__ == "__main__":
    unittest.main()
