from __future__ import annotations

import math
import sqlite3
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_ledger as ledger  # noqa: E402
from analyzing_llm_rationale import fund_vix_fade as vf  # noqa: E402

MON, TUE, WED = date(2026, 11, 9), date(2026, 11, 10), date(2026, 11, 11)


class SignalTests(unittest.TestCase):
    def test_a_sharp_fall_in_vix_before_the_session_is_a_short_and_a_jump_a_long(self):
        later = (TUE, 99.0)  # any close dated on or after the session shows Monday's is final
        vix = [(MON - timedelta(days=3), 20.0), (MON, 17.0), later]  # Friday to Monday: -3 points
        self.assertEqual(vf.signal(vix, TUE)["side"], "SHORT")
        self.assertEqual(vf.signal([(MON - timedelta(days=3), 17.0), (MON, 20.0), later], TUE)["side"], "LONG")
        self.assertEqual(vf.signal([(MON - timedelta(days=3), 17.0), (MON, 17.5), later], TUE)["side"], "NONE")

    def test_only_closes_dated_before_the_session_count_and_a_missing_close_waits(self):
        vix = [(MON - timedelta(days=3), 20.0), (MON, 17.0), (TUE, 40.0)]
        self.assertEqual(vf.signal(vix, TUE)["vix_date"], MON.isoformat())  # Tuesday's own close is not used
        self.assertIsNone(vf.signal([(MON - timedelta(days=3), 20.0), (MON, 17.0)], TUE))  # Tuesday not posted yet
        stale = [(MON - timedelta(days=10), 20.0), (MON - timedelta(days=9), 17.0), (TUE, 18.0)]
        self.assertIsNone(vf.signal(stale, TUE))

    def test_a_session_is_final_only_after_its_close(self):
        self.assertEqual(vf.finished_through(datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)), MON)  # 14:30 IST
        self.assertEqual(vf.finished_through(datetime(2026, 11, 10, 10, 30, tzinfo=timezone.utc)), TUE)  # 16:00 IST


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        ledger.ensure_schema(self.conn)
        self.now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)

    def test_each_session_is_written_once_with_its_result_after_costs(self):
        vix = [(MON - timedelta(days=3), 20.0), (MON, 17.0), (TUE, 17.2)]  # FRED has Tuesday, not Wednesday
        bank = [(MON, 59_000.0, 59_100.0), (TUE, 60_000.0, 59_700.0), (WED, 59_700.0, 59_800.0)]
        # Monday is before the test began; Wednesday waits until FRED proves Tuesday is final.
        self.assertEqual(vf.record(self.conn, "f", self.now, vix, bank), 1)
        s = vf.summary(self.conn, "f")
        self.assertEqual((s["sessions"], s["shorts"], s["verdict"]), (1, 1, "pending"))
        self.assertAlmostEqual(s["short_mean_return"], round(-math.log(59_700 / 60_000) - vf.COST, 6))
        vix.append((WED, 17.4))
        self.assertEqual(vf.record(self.conn, "f", self.now, vix, bank), 1)  # Wednesday, once FRED is past it
        self.assertEqual(vf.record(self.conn, "f", self.now, vix, bank), 0)  # never twice
        self.assertEqual(vf.summary(self.conn, "f")["sessions"], 2)


if __name__ == "__main__":
    unittest.main()
