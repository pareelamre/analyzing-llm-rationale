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


def _setup(monkey_store):
    client = _Client()
    client.store.update(monkey_store)
    server._get_datastore = lambda: client
    # inject a fake google.cloud.datastore so `_ds.Entity(...)` works offline
    sys.modules["google.cloud.datastore"] = types.SimpleNamespace(
        Entity=lambda key=None, exclude_from_indexes=(): _Entity(key=key))
    return client


class AuthDedupTests(unittest.TestCase):
    """A TestCase so CI runs these at all.

    They were module-level pytest functions, and CI runs
    `python -m unittest discover`, which only collects TestCase subclasses --
    so these two assertions had never executed there. Being a TestCase also
    gives them a tearDown, which is what _setup was missing: it reassigned
    server._get_datastore and installed a fake google.cloud.datastore into
    sys.modules, and nothing put either back. The fake outlived this module
    and broke tests/test_rag.py, whose _rag_add calls
    client.key("User", user_id, "VectorChunk") against a _Client.key that
    takes two arguments.
    """

    def setUp(self):
        self._original_getter = server._get_datastore
        self._had_datastore_module = "google.cloud.datastore" in sys.modules
        self._original_datastore_module = sys.modules.get("google.cloud.datastore")

    def tearDown(self):
        server._get_datastore = self._original_getter
        # Distinguish "was absent" from "was present": setting a missing entry
        # to None would make a later real import fail confusingly.
        if self._had_datastore_module:
            sys.modules["google.cloud.datastore"] = self._original_datastore_module
        else:
            sys.modules.pop("google.cloud.datastore", None)

    def test_new_email_creates_account_keyed_by_sub(self):
        client = _setup({})
        uid = server._upsert_user("sub-A", "a@x.com", "A", "")
        self.assertEqual(uid, "sub-A")
        self.assertEqual(len(client.store), 1)

    def test_same_email_new_sub_reuses_canonical_not_duplicate(self):
        # existing account for the email, under an older sub
        import datetime
        old = _Entity(key=_Key("sub-OLD"))
        old.update(email="dup@x.com", name="Old", created_at=datetime.datetime(2020, 1, 1))
        client = _setup({"sub-OLD": old})

        uid = server._upsert_user("sub-NEW", "dup@x.com", "New Name", "pic")

        # resolves to the canonical (oldest) account, does NOT fork a duplicate
        self.assertEqual(uid, "sub-OLD")
        self.assertEqual(len(client.store), 1)
        self.assertIn("sub-NEW", client.store["sub-OLD"].get("alt_subs", []))


if __name__ == "__main__":
    unittest.main()
