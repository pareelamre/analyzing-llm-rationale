from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_record_due.py"
_SPEC = importlib.util.spec_from_file_location("track_record_due_test_module", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
due = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(due)

NOW = datetime(2026, 11, 16, 7, 0, tzinfo=timezone.utc)


class TrackRecordDueTests(unittest.TestCase):
    def test_a_record_younger_than_fifty_minutes_waits(self):
        made = (NOW - timedelta(minutes=20)).isoformat()
        self.assertFalse(due.is_due(made, NOW))

    def test_a_record_fifty_minutes_old_is_refreshed(self):
        self.assertTrue(due.is_due((NOW - timedelta(minutes=50)).isoformat(), NOW))
        self.assertTrue(due.is_due("2026-11-16T02:03:09.112611+00:00", NOW))

    def test_a_missing_or_unreadable_timestamp_is_refreshed(self):
        self.assertTrue(due.is_due(None, NOW))
        self.assertTrue(due.is_due("not a time", NOW))

    def test_a_naive_timestamp_reads_as_utc(self):
        self.assertFalse(due.is_due("2026-11-16T06:30:00", NOW))

    def _run(self, payload, event="workflow_run"):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "track_record_live.json"
            if payload is not None:
                path.write_text(json.dumps(payload), encoding="utf-8")
            out = io.StringIO()
            with (
                mock.patch.dict(os.environ, {"GITHUB_EVENT_NAME": event}),
                redirect_stdout(out),
                mock.patch("sys.stderr", io.StringIO()),
            ):
                self.assertEqual(due.main(["track_record_due.py", str(path)]), 0)
            return out.getvalue().strip()

    def test_main_prints_a_github_output_line(self):
        fresh = {"generated_at": datetime.now(timezone.utc).isoformat()}
        self.assertEqual(self._run(fresh), "run=false")
        self.assertEqual(self._run({"generated_at": "2026-11-16T02:03:09+00:00"}), "run=true")
        self.assertEqual(self._run(None), "run=true")

    def test_a_manual_dispatch_always_runs(self):
        fresh = {"generated_at": datetime.now(timezone.utc).isoformat()}
        self.assertEqual(self._run(fresh, event="workflow_dispatch"), "run=true")


if __name__ == "__main__":
    unittest.main()
