from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_news_corpus", _ROOT / "scripts" / "fund_news_corpus.py")
nc = importlib.util.module_from_spec(_spec)
sys.modules["fund_news_corpus"] = nc
_spec.loader.exec_module(nc)


class HeadlineTests(unittest.TestCase):
    def test_only_the_first_line_is_kept_without_bloombergs_dashes(self):
        self.assertEqual(nc.headline("-- Oil Prices Climb on OPEC Cut\n\nOil rose on Friday..."),
                         "Oil Prices Climb on OPEC Cut")
        self.assertEqual(nc.headline("Dow ends higher\n\nbody"), "Dow ends higher")

    def test_a_day_level_stamp_is_seen_at_the_end_of_its_day(self):
        self.assertEqual(nc.seen_at("2017-01-03T00:00:00Z").isoformat(), "2017-01-04T00:00:00+00:00")
        self.assertEqual(nc.seen_at("2017-01-03T15:49:13Z").isoformat(), "2017-01-03T15:49:13+00:00")

    def test_groups_follow_the_pre_registered_terms(self):
        self.assertEqual(nc.groups_for("cnbc_headlines", "Wall Street slips as oil prices jump"),
                         ["us_equity", "crude"])
        self.assertEqual(nc.groups_for("cnbc_headlines", "Sensex hits a record"), ["india_equity"])
        self.assertEqual(nc.groups_for("cnbc_headlines", "Apple unveils a new phone"), [])
        self.assertEqual(nc.groups_for("cnbc_headlines", "Brentwood bakery expands"), [])
        # Every S&P 500 daily headline is US equities, whatever it says.
        self.assertEqual(nc.groups_for("sp500_daily_headlines", "Apple unveils a new phone"), ["us_equity"])


class BuildTests(unittest.TestCase):
    def test_build_writes_sorted_groups_and_counts_a_same_day_repeat_once(self):
        rows = {
            "bloomberg_reuters": [
                ("2010-05-04T14:00:00Z", "-- Crude Oil Falls\n\nbody"),
                ("2010-05-04T09:00:00Z", "-- Crude Oil Falls\n\nearlier copy"),
                ("2010-05-05T09:00:00Z", "-- Crude Oil Falls\n\nnext day"),
                ("2010-05-04T10:00:00Z", "-- Apple Profit Rises\n\nbody"),
                ("not a date", "-- OPEC meets\n\nbody"),
            ],
            "sp500_daily_headlines": [("2010-05-04T00:00:00Z", "Investors weigh earnings")],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            coverage = nc.build(rows, out)
            crude = [json.loads(line) for line in (out / "crude.jsonl").read_text().splitlines()]
            us = [json.loads(line) for line in (out / "us_equity.jsonl").read_text().splitlines()]
            self.assertEqual([c["seen"] for c in crude], ["2010-05-04T09:00:00+00:00", "2010-05-05T09:00:00+00:00"])
            self.assertEqual(us, [{"seen": "2010-05-05T00:00:00+00:00", "title": "Investors weigh earnings"}])
            self.assertEqual((out / "india_equity.jsonl").read_text(), "")
            self.assertEqual(coverage["subsets"]["bloomberg_reuters"], {
                "rows": 5, "matched": 3, "first": "2010-05-04T09:00:00+00:00", "last": "2010-05-05T09:00:00+00:00"})
            self.assertEqual(coverage["groups"]["crude"]["headlines"], 2)
            self.assertTrue((out / "_coverage.json").is_file())


if __name__ == "__main__":
    unittest.main()
