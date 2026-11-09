from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import duckdb

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fund_rationale_check", _ROOT / "scripts" / "fund_rationale_check.py")
check = importlib.util.module_from_spec(_spec)
sys.modules["fund_rationale_check"] = check
_spec.loader.exec_module(check)

NOW = datetime(2026, 11, 9, 0, 0, tzinfo=timezone.utc)


def _store(path: Path, rows) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE forecast_snapshot (key VARCHAR, platform VARCHAR, ident VARCHAR, model VARCHAR,"
                " question VARCHAR, model_probability DOUBLE, snapshot_ts VARCHAR, rationale VARCHAR)")
    con.executemany("INSERT INTO forecast_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


ROWS = [
    # The inverted forecast of 2026-11-08: the rationale argues No, the number says 98% Yes.
    ("k1", "Polymarket", "iran", check.FORECAST_MODEL, "Will the Iranian regime fall before 2027?", 0.98,
     "2026-11-08T17:25:16+00:00", "Intelligence assessments do not guarantee the regime will be dissolved..."),
    ("k1b", "Polymarket", "iran", check.FORECAST_MODEL, "Will the Iranian regime fall before 2027?", 0.97,
     "2026-11-07T17:25:16+00:00", "older snapshot of the same market"),
    ("k2", "Kalshi", "KXFED", check.FORECAST_MODEL, "Will the Fed cut?", 0.30,
     "2026-11-08T17:26:00+00:00", "Evidence points to a hold."),
    ("k3", "Kalshi", "OLD", check.FORECAST_MODEL, "Too old?", 0.5, "2026-11-01T00:00:00+00:00", "stale"),
    ("k4", "Kalshi", "OTHER", "council", "Another model's forecast", 0.5, "2026-11-08T17:00:00+00:00", "x"),
    ("k5", "Kalshi", "BLANK", check.FORECAST_MODEL, "No rationale", 0.5, "2026-11-08T17:00:00+00:00", "  "),
]


class RationaleCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = self.tmp / "store.duckdb"
        _store(self.store, ROWS)

    def test_only_the_latest_recent_forecast_per_market_with_a_rationale_is_pending(self):
        pending = check.pending_forecasts(self.store, set(), now=NOW)
        self.assertEqual([p["key"] for p in pending], ["k1", "k2"])
        self.assertEqual(pending[0]["instrument_id"], "polymarket:iran")
        self.assertEqual(check.pending_forecasts(self.store, {"k1", "k2"}, now=NOW), [])

    def test_a_number_pointing_away_from_its_own_rationale_is_flagged(self):
        self.assertEqual(check.verdict(0.98, 0.04), "contradicts")
        self.assertEqual(check.verdict(0.02, 0.04), "agrees")
        self.assertEqual(check.verdict(0.30, 0.85), "contradicts")
        self.assertEqual(check.verdict(0.80, 0.85), "agrees")
        self.assertEqual(check.verdict(0.98, 0.5), "unclear")

    def test_a_run_appends_verdicts_and_never_rechecks(self):
        out = self.tmp / "checks.json"
        asked = []

        def fake_checker(question, rationale):
            asked.append(question)
            return 0.05 if "Iranian" in question else 0.2

        first = check.run(self.store, out, fake_checker, now=NOW)
        second = check.run(self.store, out, fake_checker, now=NOW)
        self.assertEqual((first["checked"], second["checked"]), (2, 0))
        self.assertEqual(len(asked), 2)
        payload = json.loads(out.read_text())
        by_key = {c["key"]: c for c in payload["checks"]}
        self.assertEqual(by_key["k1"]["verdict"], "contradicts")
        self.assertEqual(by_key["k2"]["verdict"], "agrees")
        self.assertEqual(payload["mode"], "shadow")
        self.assertEqual(payload["checker"]["model"], "ZefanCai/Open-Jev-2B")


class ValidationTests(unittest.TestCase):
    def test_the_fixture_holds_the_four_inversions_and_scoring_counts_them(self):
        cases = _ROOT / "tests" / "fixtures" / "rationale_check_cases.json"
        data = json.loads(cases.read_text())
        self.assertEqual(sum(c["expected"] == "contradicts" for c in data["cases"]), 4)
        # A checker that reads every rationale as arguing "No" catches every inversion here,
        # because all four inverted numbers point to Yes, but it would also miss Mojtaba (agrees).
        result = check.validate(cases, lambda q, r: 0.05)
        self.assertEqual(result["summary"]["inversions_caught"], "4/4")
        self.assertEqual(result["summary"]["false_flags"], 1)


if __name__ == "__main__":
    unittest.main()
