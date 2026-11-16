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
_spec = importlib.util.spec_from_file_location("fund_rationale_checker_eval",
                                               _ROOT / "scripts" / "fund_rationale_checker_eval.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["fund_rationale_checker_eval"] = ev
_spec.loader.exec_module(ev)

MODEL = ev.check.FORECAST_MODEL


def _store(path: Path, rows) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE forecast_snapshot (key VARCHAR, platform VARCHAR, ident VARCHAR, model VARCHAR,"
                " question VARCHAR, model_probability DOUBLE, snapshot_ts VARCHAR, rationale VARCHAR)")
    con.executemany("INSERT INTO forecast_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()


class CasesTests(unittest.TestCase):
    def test_live_cases_are_clear_post_fix_forecasts_counted_once_per_rationale(self):
        rows = [
            ("a", "Kalshi", "FED", MODEL, "Fed cut?", 0.85, "2026-11-10T01:00:00+00:00", "Likely a cut."),
            ("a2", "Kalshi", "FED", MODEL, "Fed cut?", 0.86, "2026-11-11T01:00:00+00:00", "Likely a cut."),
            ("b", "Polymarket", "ham", MODEL, "Hamas?", 0.05, "2026-11-12 05:00:00+00", "Very unlikely."),
            ("c", "Kalshi", "MID", MODEL, "Coin flip?", 0.55, "2026-11-12T01:00:00+00:00", "Could go either way."),
            ("d", "Kalshi", "OLD", MODEL, "Before the fix?", 0.98, "2026-11-08T17:25:00+00:00", "No."),
            ("e", "Kalshi", "LATE", MODEL, "After the window?", 0.98, "2026-11-16T00:00:00+00:00", "Yes."),
            ("f", "Kalshi", "OTHER", "council", "Other model", 0.9, "2026-11-10T01:00:00+00:00", "Yes."),
            ("f2", "Kalshi", "OTHER", "council", "Other model", 0.1, "2026-11-11T01:00:00+00:00", "Now no."),
            ("f3", "Kalshi", "OTHER", "gpt-oss-120b", "Other model", 0.5, "2026-11-11T01:00:00+00:00", "Unsure."),
            ("f4", "Kalshi", "FED", "gpt-oss-120b", "Fed cut?", 0.95, "2026-11-12T01:00:00+00:00", "A cut."),
            ("g", "Kalshi", "BLANK", MODEL, "No rationale", 0.9, "2026-11-10T01:00:00+00:00", "  "),
            ("h", "Kalshi", "EDGE", MODEL, "Exactly at the bar", 0.30, "2026-11-13T01:00:00+00:00", "Unlikely."),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "s.duckdb"
            _store(store, rows)
            cases = ev.live_cases(store)
        set_a = [c for c in cases if c["set"] == "A"]
        set_b = [c for c in cases if c["set"] == "B"]
        self.assertEqual([c["id"] for c in set_a], ["live:a", "live:b", "live:h"])
        self.assertEqual([c["label"] for c in set_a], [1, 0, 0])
        self.assertEqual(set_a[1]["market"], "polymarket:ham")
        # Set B (amendment): each other model's latest clear forecast per market, after set A.
        self.assertEqual(sorted(c["id"] for c in set_b), ["live:f2", "live:f4"])
        self.assertEqual(cases[:3], set_a)
        self.assertEqual({c["id"]: c["label"] for c in set_b}, {"live:f2": 0, "live:f4": 1})

    def test_known_cases_are_the_seven_of_2026_11_08_with_four_inversions(self):
        known = ev.known_cases()
        self.assertEqual(len(known), 7)
        self.assertEqual(sum(c["expected"] == "contradicts" for c in known), 4)

    def test_no_new_case_starts_after_the_time_budget(self):
        ticks = iter(range(0, 1000, 40))
        cases = [{"id": str(i), "question": "q", "rationale": "r", "forecast": 0.9} for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            out = ev.score(cases, lambda q, r: 0.9, "clm", Path(tmp) / "s.json", budget_seconds=100,
                           clock=lambda: next(ticks))
        self.assertTrue(out["stopped_early"])
        self.assertEqual(len(out["scores"]), 2)

    def test_shards_split_the_cases_without_overlap(self):
        cases = [{"id": str(i)} for i in range(13)]
        got = [c["id"] for i in range(6) for c in ev.shard_of(cases, f"{i}/6")]
        self.assertEqual(sorted(got, key=int), [str(i) for i in range(13)])


def _cases(n_live: int, n_markets: int) -> list[dict]:
    known = ev.known_cases()
    live = [{"id": f"live:{i}", "kind": "live", "set": "AB"[i % 2], "model": "m", "market": f"m{i % n_markets}",
             "forecast": 0.9 if i % 2 else 0.1,
             "label": i % 2, "question": "q", "rationale": "r"} for i in range(n_live)]
    return known + live


def _scores(cases, p_for) -> dict:
    return {c["id"]: {"p_yes": p_for(c), "seconds": 5.0} for c in cases}


def _right(c):
    """A checker that reads every rationale correctly."""
    if c["kind"] == "known":
        inverted = c["expected"] == "contradicts"
        return 0.1 if (c["forecast"] > 0.5) == inverted else 0.9
    return 0.95 if c["label"] else 0.05


class ReportTests(unittest.TestCase):
    def test_a_clearly_better_and_fast_clm_switches(self):
        cases = _cases(60, 20)
        scores = {"clm": _scores(cases, _right),
                  "jev2b": _scores(cases, lambda c: 0.6 if c["kind"] == "live" else _right(c))}
        r = ev.report(cases, scores)
        self.assertEqual(r["verdict"], "SWITCH")
        self.assertLess(r["brier_difference_ci95"][1], 0)
        self.assertEqual(r["checkers"]["clm"]["known"]["inversions_caught"], 4)
        self.assertEqual(r["checkers"]["clm"]["known"]["false_flags"], 0)
        self.assertEqual(r["description_only"]["by_set"]["A"]["cases"] + r["description_only"]["by_set"]["B"]["cases"], 60)

    def test_a_worse_or_slow_clm_keeps_the_2b(self):
        cases = _cases(60, 20)
        worse = {"clm": _scores(cases, lambda c: 0.6 if c["kind"] == "live" else _right(c)),
                 "jev2b": _scores(cases, _right)}
        self.assertEqual(ev.report(cases, worse)["verdict"], "KEEP")
        slow = {"clm": {k: {**v, "seconds": 90.0} for k, v in _scores(cases, _right).items()},
                "jev2b": _scores(cases, lambda c: 0.6 if c["kind"] == "live" else _right(c))}
        r = ev.report(cases, slow)
        self.assertEqual(r["verdict"], "KEEP")
        self.assertFalse(r["tests"]["fast_enough"])

    def test_missing_an_inversion_keeps_the_2b(self):
        cases = _cases(60, 20)
        miss = lambda c: 0.5 if c["kind"] == "known" and c["expected"] == "contradicts" else _right(c)  # noqa: E731
        r = ev.report(cases, {"clm": _scores(cases, miss),
                              "jev2b": _scores(cases, lambda c: 0.6 if c["kind"] == "live" else _right(c))})
        self.assertEqual(r["verdict"], "KEEP")
        self.assertFalse(r["tests"]["known_cases_no_worse"])

    def test_too_few_live_cases_or_markets_keeps_the_2b(self):
        for n_live, n_markets in ((30, 20), (60, 10)):
            cases = _cases(n_live, n_markets)
            r = ev.report(cases, {"clm": _scores(cases, _right),
                                  "jev2b": _scores(cases, lambda c: 0.6 if c["kind"] == "live" else _right(c))})
            self.assertEqual(r["verdict"], "KEEP (too few to tell)")

    def test_the_bootstrap_resamples_whole_markets(self):
        lo, hi = ev.market_bootstrap([("a", 1.0), ("a", 1.0), ("b", -1.0)], draws=2000)
        self.assertLessEqual(lo, -1.0 + 1e-9)
        self.assertGreaterEqual(hi, 1.0 - 1e-9)

    def test_the_report_holds_no_rationale_text(self):
        cases = _cases(60, 20)
        r = ev.report(cases, {"clm": _scores(cases, _right), "jev2b": _scores(cases, _right)})
        self.assertNotIn("rationale", json.dumps(r["cases"]))


if __name__ == "__main__":
    unittest.main()
