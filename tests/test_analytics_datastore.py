"""Datastore-backed analytics: cumulative counts must survive across requests
(i.e. not depend on the ephemeral DuckDB), dedupe unique visitors by a stable
id, and produce a per-day breakdown. Uses an in-memory fake Datastore client."""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_ds_analytics.duckdb")
os.environ["ENABLE_OTEL"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server  # noqa: E402
from analyzing_llm_rationale.server import _ANALYTICS_DB, app  # noqa: E402


class FakeEntity(dict):
    def __init__(self, key=None, exclude_from_indexes=()):  # noqa: D401
        super().__init__()
        self.key = key


class FakeKey:
    def __init__(self, kind, id_=None):
        self.kind = kind
        self.id = id_


class FakeClient:
    """Just enough of google.cloud.datastore.Client for the analytics path."""

    def __init__(self):
        self.store = {}
        self._auto = 0

    def key(self, kind, id_=None):
        return FakeKey(kind, id_)

    def get(self, key):
        return self.store.get((key.kind, key.id))

    def put(self, entity):
        key = entity.key
        if key.id is None:
            self._auto += 1
            key.id = self._auto
        self.store[(key.kind, key.id)] = entity

    def transaction(self):
        return contextlib.nullcontext()

    def query(self, kind):
        return _FakeQuery(self, kind)


class _FakeQuery:
    def __init__(self, client, kind):
        self.client = client
        self.kind = kind

    def add_filter(self, *args, **kwargs):
        return None  # window filter irrelevant for the test's fresh data

    def fetch(self):
        return [e for (k, _id), e in self.client.store.items() if k == self.kind]


class DatastoreAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeClient()
        fake_module = types.ModuleType("google.cloud.datastore")
        fake_module.Entity = FakeEntity
        self.patches = [
            mock.patch.object(server, "_get_datastore", return_value=self.fake),
            mock.patch.dict(sys.modules, {"google.cloud.datastore": fake_module}),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app)

    def tearDown(self):
        for p in self.patches:
            p.stop()
        if _ANALYTICS_DB.exists():
            _ANALYTICS_DB.unlink()

    def _visit(self, path, ua, ip):
        return self.client.post(
            "/analytics/visit",
            json={"path": path, "referrer": "", "timezone": "UTC"},
            headers={"user-agent": ua, "x-forwarded-for": ip},
        )

    def test_cumulative_counts_persist_and_dedupe_unique_visitors(self):
        self._visit("/", "UA-one", "1.1.1.1")
        self._visit("/about", "UA-one", "1.1.1.1")   # same person, 2nd visit
        self._visit("/", "UA-two", "2.2.2.2")        # different person

        summary = self.client.get("/analytics/summary").json()
        self.assertEqual(summary["total_visits"], 3)
        self.assertEqual(summary["unique_visitors"], 2)
        self.assertEqual(summary["by_day"][0]["visits"], 3)
        self.assertEqual(summary["by_day"][0]["unique_visitors"], 2)

    def test_no_raw_ip_persisted(self):
        self._visit("/", "UA-one", "9.9.9.9")
        for entity in self.fake.store.values():
            self.assertNotIn("9.9.9.9", repr(dict(entity)))

    def test_counts_independent_of_local_duckdb(self):
        # DuckDB starts empty; counts must come from Datastore, proving the
        # number survives a fresh (recycled) instance with no local DB.
        self._visit("/", "UA-x", "3.3.3.3")
        self.assertFalse(_ANALYTICS_DB.exists())  # Datastore path never touched DuckDB
        summary = self.client.get("/analytics/summary").json()
        self.assertEqual(summary["total_visits"], 1)


if __name__ == "__main__":
    unittest.main()
