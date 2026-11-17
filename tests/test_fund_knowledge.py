"""The knowledge base's view on the fund's commodity pairs (made-up forecasts, no licensed data)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import fund_knowledge as fk  # noqa: E402
from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale.fund_orchestrator import Signal  # noqa: E402

OLD, NEW = "acme-2026-04-metals", "acme-2026-10-metals"
PAIRS = [
    {"key": "copper-aluminium", "a": {"id": "mcx:COPPER", "underlying": "copper"},
     "b": {"id": "mcx:ALUMINI", "underlying": "aluminium"}},
    {"key": "gold-silver", "a": {"id": "mcx:GOLDPETAL", "underlying": "gold"},
     "b": {"id": "mcx:SILVERMIC", "underlying": "silver"}},
    {"key": "gold-mcx-kalshi", "a": {"id": "mcx:GOLDPETAL", "underlying": "gold"},
     "b": {"id": "kalshi_perp:GOLD", "underlying": "gold"}},
]


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows) if isinstance(rows, dict) else "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")


def price(sid, n, metric, value, year, scenario="Acme baseline"):
    return {"id": f"{sid}:c{n:03d}", "kind": "projection", "text": f"{metric} {value} in {year}.",
            "entities": [], "metric": metric, "value": value, "unit": "$/t", "geo": "global",
            "horizon": str(year), "scenario": scenario}


def make_base(root: Path) -> Path:
    _write(root / "entities.jsonl", [{"id": "copper", "name": "Copper", "type": "commodity", "aliases": []}])
    for sid, published, claims in (
        (OLD, "2026-04-01", [price(OLD, 1, "copper price", 10000, 2026), price(OLD, 2, "copper price", 9000, 2027),
                             price(OLD, 3, "aluminium price", 2500, 2026),
                             price(OLD, 4, "aluminium price", 2500, 2027)]),
        (NEW, "2026-10-01", [price(NEW, 1, "copper price", 12000, 2026), price(NEW, 2, "copper price", 13000, 2027),
                             price(NEW, 3, "aluminium price", 3000, 2026),
                             price(NEW, 4, "aluminium price", 3000, 2027),
                             price(NEW, 5, "copper price", 6000, 2027, scenario="Acme adverse scenario"),
                             price(NEW, 6, "gold price", 4000, 2026), price(NEW, 7, "gold price", 4200, 2027),
                             price(NEW, 8, "silver price", 60, 2026)]),
    ):
        _write(root / "sources" / sid / "source.json", {"id": sid, "title": "Metals Outlook", "publisher": "Acme",
                                                        "published": published})
        _write(root / "sources" / sid / "claims.jsonl", claims)
    return root


def legs(side_a: str) -> list[Signal]:
    pair = PAIRS[0]
    side_b = "SHORT" if side_a == "LONG" else "LONG"
    shared = [{"label": "pair", "value": "Copper / Aluminium", "source": "fund universe"}]
    return [Signal(analyst="pairs_model", instrument=dict(pair["a"]), side=side_a, price=0.0, edge=0.003, trust=1.0,
                   facts=shared, pair_key=pair["key"]),
            Signal(analyst="pairs_model", instrument=dict(pair["b"]), side=side_b, price=0.0, edge=0.003, trust=1.0,
                   facts=shared, pair_key=pair["key"])]


class PairViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = kb.Index(kb.load(make_base(Path(self.tmp.name))))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_newest_baseline_source_sets_the_direction(self):
        view = fk.pair_view(self.index, "copper", "aluminium", date(2026, 11, 17))
        self.assertEqual(view["source"], NEW)  # the adverse 6,000 is ignored
        self.assertEqual(view["ratio"], [4.0, 4.3333])
        self.assertEqual(view["favours"], "a")
        self.assertEqual(view["cite"], "Acme, Oct 2026")

    def test_before_the_newer_report_the_older_one_is_the_view(self):
        view = fk.pair_view(self.index, "copper", "aluminium", date(2026, 6, 1))
        self.assertEqual(view["source"], OLD)
        self.assertEqual(view["favours"], "b")  # 4.0 -> 3.6

    def test_no_view_without_both_years_or_for_a_same_metal_pair(self):
        self.assertIsNone(fk.pair_view(self.index, "gold", "silver", date(2026, 11, 17)))
        self.assertIsNone(fk.pair_view(self.index, "gold", "gold", date(2026, 11, 17)))
        self.assertIsNone(fk.pair_view(self.index, "copper", "unobtainium", date(2026, 11, 17)))

    def test_each_leg_gets_the_view_and_whether_it_agrees(self):
        signals = legs("LONG")
        original = signals[0].facts
        views = fk.annotate(signals, PAIRS, date(2026, 11, 17), index=self.index)
        for s in signals:
            self.assertEqual(s.facts[-1]["label"], "knowledge base view")
            self.assertIn("agrees with buying copper", s.facts[-1]["value"])
            self.assertIn("from 4.00 (2026) to 4.33 (2027), +8.3%", s.facts[-1]["value"])
        self.assertEqual(len(original), 1)  # the shared list is not mutated
        self.assertTrue(views["copper-aluminium"]["agrees"])
        self.assertTrue(views["copper-aluminium"]["traded"])

    def test_a_trade_against_the_view_is_marked(self):
        signals = legs("SHORT")
        views = fk.annotate(signals, PAIRS, date(2026, 11, 17), index=self.index)
        self.assertIn("against buying aluminium", signals[0].facts[-1]["value"])
        self.assertFalse(views["copper-aluminium"]["agrees"])

    def test_untraded_pairs_still_get_a_view_for_the_page(self):
        views = fk.annotate([], PAIRS, date(2026, 11, 17), index=self.index)
        self.assertEqual(set(views), {"copper-aluminium"})
        self.assertFalse(views["copper-aluminium"]["traded"])
        self.assertIsNone(views["copper-aluminium"]["agrees"])

    def test_without_a_base_nothing_changes(self):
        kb._shared.clear()
        signals = legs("LONG")
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": ""}):
            self.assertEqual(fk.annotate(signals, PAIRS, date(2026, 11, 17)), {})
        self.assertEqual(len(signals[0].facts), 1)


if __name__ == "__main__":
    unittest.main()
