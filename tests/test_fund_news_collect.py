from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_news_collect", _ROOT / "scripts" / "fund_news_collect.py")
nc = importlib.util.module_from_spec(_spec)
sys.modules["fund_news_collect"] = nc
_spec.loader.exec_module(nc)

NOW = datetime(2026, 11, 9, 6, 0, tzinfo=timezone.utc)


class CollectTests(unittest.TestCase):
    def test_a_deadline_keeps_recent_windows_for_every_group(self):
        ticks = iter(range(0, 10_000, 30))  # every clock read advances 30 s

        def fetcher(query, start, end, session):
            stamp = end.strftime("%Y%m%dT%H%M%SZ")
            return [{"url": f"{query}-{stamp}", "title": f"headline {stamp}", "seendate": stamp}]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            coverage = nc.collect(3, out, now=NOW, deadline_minutes=2, fetcher=fetcher,
                                  sleep=lambda s: None, clock=lambda: next(ticks))
            self.assertTrue(coverage["deadline_hit"])
            self.assertEqual(coverage["windows_planned"], 6)
            # Groups take turns, so the deadline cut every group at the same recent window.
            self.assertEqual(len(set(coverage["windows_done"].values())), 1)
            self.assertGreater(min(coverage["windows_done"].values()), 0)
            newest = json.loads((out / "gold.jsonl").read_text().splitlines()[0])
            self.assertEqual(newest["seen"], "2026-11-09T06:00:00+00:00")
            self.assertTrue((out / "_coverage.json").is_file())

    def test_a_window_that_keeps_failing_is_skipped_not_fatal(self):
        def fetcher(query, start, end, session):
            raise RuntimeError("GDELT kept rate-limiting")

        with tempfile.TemporaryDirectory() as tmp:
            coverage = nc.collect(1, Path(tmp), now=NOW, fetcher=fetcher, sleep=lambda s: None, clock=lambda: 0)
        self.assertEqual(coverage["windows_done"], {g: 0 for g in nc.GROUPS})
        self.assertFalse(coverage["deadline_hit"])


if __name__ == "__main__":
    unittest.main()
