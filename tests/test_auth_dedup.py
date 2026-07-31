from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import server  # noqa: E402


class _Key:
    def __init__(self, name):
        self.name = name


class _Entity(dict):
    def __init__(self, key=None, **_):
        super().__init__()
        self.key = key


class _Query:
    def __init__(self, store):
        self._store = store
        self._email = None

    def add_filter(self, prop, op, val):
        if prop == "email":
            self._email = val

    def fetch(self):
        return [e for e in self._store.values() if e.get("email") == self._email]


class _Client:
    def __init__(self):
        self.store = {}

    def key(self, kind, name):
        return _Key(name)

    def get(self, key):
        return self.store.get(key.name)

    def query(self, kind=None):
        return _Query(self.store)

    def put(self, entity):
        self.store[entity.key.name] = entity


class AuthDedupTests(unittest.TestCase):
    def setUp(self):
        self._orig_get_datastore = server._get_datastore
        self._orig_ds_module = sys.modules.get("google.cloud.datastore")

    def tearDown(self):
        server._get_datastore = self._orig_get_datastore
        if self._orig_ds_module is None:
            sys.modules.pop("google.cloud.datastore", None)
        else:
            sys.modules["google.cloud.datastore"] = self._orig_ds_module

    def _setup(self, monkey_store):
        client = _Client()
        client.store.update(monkey_store)
        server._get_datastore = lambda: client
        # inject a fake google.cloud.datastore so `_ds.Entity(...)` works offline
        sys.modules["google.cloud.datastore"] = types.SimpleNamespace(
            Entity=lambda key=None, exclude_from_indexes=(): _Entity(key=key))
        return client

    def test_new_email_creates_account_keyed_by_sub(self):
        client = self._setup({})
        uid = server._upsert_user("sub-A", "a@x.com", "A", "")
        assert uid == "sub-A"
        assert len(client.store) == 1

    def test_same_email_new_sub_reuses_canonical_not_duplicate(self):
        import datetime
        old = _Entity(key=_Key("sub-OLD"))
        old.update(email="dup@x.com", name="Old", created_at=datetime.datetime(2020, 1, 1))
        client = self._setup({"sub-OLD": old})

        uid = server._upsert_user("sub-NEW", "dup@x.com", "New Name", "pic")

        # resolves to the canonical (oldest) account, does NOT fork a duplicate
        assert uid == "sub-OLD"
        assert len(client.store) == 1
        assert "sub-NEW" in client.store["sub-OLD"].get("alt_subs", [])


if __name__ == "__main__":
    unittest.main()
