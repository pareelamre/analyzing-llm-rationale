"""An immutable intent must survive the same durable reservation as its command."""

import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.store import DatastoreTwinStore, InMemoryTwinStore, TwinStoreError
from tests.test_twin_recovery import prepared_unknown


class TwinIntentStoreTests(unittest.TestCase):
    def test_memory_restores_original_intent_only_for_its_exact_command_scope(self):
        store, intent, command, _ = prepared_unknown(store=InMemoryTwinStore())
        self.assertEqual(store.intent_for_command(command.scope_id, command.id), intent)
        with self.assertRaises(TwinStoreError):
            store.intent_for_command("another-scope", command.id)

    def test_sql_restart_restores_exact_payload_and_rejects_corruption(self):
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FORESEA_DATASTORE_BACKEND": "sql"}),
        ):
            path = str(Path(root) / "state.sqlite3")
            with closing(Client(path=path)) as first:
                _, intent, command, _ = prepared_unknown(store=DatastoreTwinStore(first))
            with closing(Client(path=path)) as second:
                store = DatastoreTwinStore(second)
                self.assertEqual(store.intent_for_command(command.scope_id, command.id), intent)
                entity = second.get(store._key(command.scope_id, "TwinIntent", command.intent_hash))
                self.assertIsNotNone(entity)
                entity["payload_json"] = entity["payload_json"].replace(
                    "recovery-intent", "tampered-intent"
                )
                second.put(entity)
                with self.assertRaises(TwinStoreError):
                    store.intent_for_command(command.scope_id, command.id)

    def test_missing_legacy_payload_does_not_invent_recovery_authority(self):
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FORESEA_DATASTORE_BACKEND": "sql"}),
            closing(
                Client(path=str(Path(root) / "state.sqlite3")),
            ) as client,
        ):
            store, _, command, _ = prepared_unknown(store=DatastoreTwinStore(client))
            client.delete(store._key(command.scope_id, "TwinIntent", command.intent_hash))
            with self.assertRaises(TwinStoreError):
                store.intent_for_command(command.scope_id, command.id)

    def test_failed_reservation_commit_never_leaves_recoverable_orphan_intent(self):
        with (
            TemporaryDirectory() as root,
            patch.dict("os.environ", {"FORESEA_DATASTORE_BACKEND": "sql"}),
            closing(
                Client(path=str(Path(root) / "state.sqlite3")),
            ) as client,
        ):
            store = DatastoreTwinStore(client)
            original = client.put_multi

            def fail_after_writes(rows):
                original(rows)
                raise RuntimeError("synthetic failed reservation transaction")

            with (
                patch.object(client, "put_multi", side_effect=fail_after_writes),
                self.assertRaises(RuntimeError),
            ):
                prepared_unknown(store=store)
            self.assertEqual(list(client.query(kind="TwinIntent").fetch()), [])
            self.assertEqual(list(client.query(kind="TwinCommand").fetch()), [])
