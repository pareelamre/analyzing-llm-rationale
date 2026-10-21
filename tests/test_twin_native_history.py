"""Private cursor continuation never substitutes partial history for inventory."""

import unittest
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analyzing_llm_rationale.datastore_sql import Client
from analyzing_llm_rationale.twin.native_history import NativeHistoryCapture, NativeHistoryPending
from analyzing_llm_rationale.twin.reconcile import AccountReadError
from tests.test_twin_recovery import NOW, prepared_unknown


class NativeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.dict("os.environ", {"FORESEA_DATASTORE_BACKEND": "sql"}))
        self.db = self.enterContext(closing(Client(path=str(Path(self.root) / "history.sqlite3"))))
        store, self.intent, self.command, _ = prepared_unknown()
        self.scope = store.account_scope(self.intent.account_scope_id)
        self.calls = []

    def capture(self, client=None, guard=lambda: None, budget=2):
        return NativeHistoryCapture(
            client or self.db,
            self.scope,
            self.command,
            guard=guard,
            clock=lambda: NOW,
            deadline=NOW + timedelta(minutes=2),
            page_budget=budget,
        )

    def fetch(self, cursor):
        page = int(cursor or "0")
        self.calls.append(page)
        return {
            "items": [{"owned": page == 24, "value": page}],
            "cursor": str(page + 1) if page < 24 else None,
        }

    def test_more_than_twenty_pages_resume_after_database_restart(self):
        while True:
            with closing(Client(path=str(Path(self.root) / "history.sqlite3"))) as db:
                capture = self.capture(db)
                try:
                    rows = capture.collection(
                        "fills", {"ticker": "fixture"}, self.fetch, select=lambda row: row["owned"]
                    )
                except NativeHistoryPending:
                    continue
                break
        self.assertEqual(rows, ({"owned": True, "value": 24},))
        self.assertEqual(self.calls, list(range(25)))

    def test_order_frame_changes_discard_old_capture_instead_of_mixing_snapshots(self):
        capture = self.capture()
        capture.select_order("native-order", {"status": "open", "filled": "0"})
        with self.assertRaises(NativeHistoryPending):
            capture.collection("fills", {}, self.fetch)
        fresh = self.capture()
        fresh.select_order("native-order", {"status": "filled", "filled": "2"})
        with self.assertRaises(NativeHistoryPending):
            fresh.collection("fills", {}, self.fetch)
        self.assertEqual(self.calls, [0, 1, 0, 1])

    def test_authority_loss_after_page_write_rolls_back_the_cursor(self):
        lost = False
        put = self.db.put

        def write_then_lose(entity):
            nonlocal lost
            put(entity)
            if entity.key.kind == "TwinNativeHistoryChunk":
                lost = True

        def guard():
            if lost:
                raise AccountReadError("fixture_lost_authority")

        capture = self.capture(guard=guard)
        with (
            patch.object(self.db, "put", side_effect=write_then_lose),
            self.assertRaises(AccountReadError),
        ):
            capture.collection("fills", {}, self.fetch)
        self.assertEqual(list(self.db.query(kind="TwinNativeHistoryChunk").fetch()), [])

    def test_cursor_repetition_across_restart_and_corrupt_pages_fail_closed(self):
        capture = self.capture(budget=1)
        with self.assertRaises(NativeHistoryPending):
            capture.collection("fills", {}, lambda _: {"items": [], "cursor": "repeat"})
        with self.assertRaises(AccountReadError):
            self.capture().collection("fills", {}, lambda _: {"items": [], "cursor": "repeat"})
        entity = next(iter(self.db.query(kind="TwinNativeHistoryChunk").fetch()))
        entity["payload_json"] += " "
        self.db.put(entity)
        with self.assertRaises(AccountReadError):
            self.capture().collection("fills", {}, self.fetch)

    def test_committed_cleanup_removes_private_pages_and_resets_next_capture(self):
        capture = self.capture(budget=32)
        capture.collection("fills", {}, self.fetch)
        with self.db.transaction():
            capture.clear_in_transaction()
        self.assertEqual(list(self.db.query(kind="TwinNativeHistoryChunk").fetch()), [])
        with self.assertRaises(NativeHistoryPending):
            self.capture().collection("fills", {}, self.fetch)
        self.assertEqual(self.calls[-2:], [0, 1])
