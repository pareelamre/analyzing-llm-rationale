"""The gap scan: which open long-horizon questions and fund pairs the knowledge base can't help with.

Uses the made-up widget base from test_knowledge plus made-up metal forecasts, so no licensed
report enters this public repo.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_gaps as gaps  # noqa: E402
from test_knowledge import _write, make_base  # noqa: E402

TODAY = date(2026, 11, 17)


def add_metals(root: Path) -> None:
    folder = root / "sources" / "acme-2026-10-metals"
    _write(folder / "source.json", {"id": "acme-2026-10-metals", "title": "Metals Outlook",
                                    "publisher": "Acme Research", "published": "2026-10-20", "summary": "Metals."})
    claims = []
    for n, (metal, values) in enumerate((("copper", (10, 11)), ("aluminium", (2.5, 2.4)))):
        for year, value in zip((2026, 2027), values):
            claims.append({"id": f"acme-2026-10-metals:c{n}{year}", "kind": "projection", "page": 2,
                           "text": f"Acme sees {metal} at {value} thousand dollars a tonne in {year}.",
                           "entities": ["acme"], "metric": f"{metal} price", "value": value,
                           "unit": "USD/t", "geo": "global", "horizon": str(year)})
    _write(folder / "claims.jsonl", claims)
    _write(folder / "relations.jsonl", [])
    _write(folder / "chunks.jsonl", [])


def snapshot(question, close, *, domain="economics", resolved=False):
    return {"question": question, "close_time": close, "domain": domain, "resolved": resolved,
            "market_url": "https://example.org/m"}


def make_foresea(root: Path) -> Path:
    _write(root / "data" / "track_record_store.json", {"ForecastSnapshot": [
        snapshot("Will widget demand pass 110 million units by 2030?", "2030-12-31T00:00:00Z"),
        snapshot("Will widget demand pass 110 million units by 2030?", "2030-12-31T00:00:00Z"),  # duplicate
        snapshot("Will a painting sell for $120 million in 2030?", "2030-12-31T00:00:00Z", domain="other"),
        snapshot("Will gadgets sell out tomorrow?", "2026-11-18T00:00:00Z"),  # too soon
        snapshot("Will the Ruritania Rovers win the 2027 cup?", "2027-06-01T00:00:00Z", domain="sports"),
        snapshot("Did widget demand pass 90 million in 2025?", "2027-01-01T00:00:00Z", resolved=True),
    ]})
    _write(root / "static" / "mark_to_market_live.json", {"edge_board": [
        {"question": "Will Ruritania's prime minister resign in 2027?", "resolve_time": "2027-12-31T00:00:00Z",
         "domain": "geopolitics", "market_url": "https://example.org/pm"}]})
    leg = lambda u, venue="mcx": {"underlying": u, "venue": venue}  # noqa: E731
    _write(root / "static" / "fund_universe.json", {"pairs": [
        {"key": "copper-aluminium", "a": leg("copper"), "b": leg("aluminium")},
        {"key": "gold-silver", "a": leg("gold"), "b": leg("silver")},
        {"key": "gold-mcx-kalshi", "a": leg("gold"), "b": leg("gold", "kalshi_perp")},
    ]})
    return root


class GapScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name) / "kb"
        make_base(base)
        add_metals(base)
        self.base = base
        self.index = kb.Index(kb.load(base))
        self.foresea = make_foresea(Path(self.tmp.name) / "foresea")

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_open_long_horizon_questions_in_scope_are_asked(self):
        rows = gaps.open_questions(self.foresea, TODAY)
        self.assertEqual(sorted(r["question"] for r in rows), [
            "Will Ruritania's prime minister resign in 2027?",
            "Will a painting sell for $120 million in 2030?",
            "Will widget demand pass 110 million units by 2030?"])
        self.assertEqual({r["origin"] for r in rows}, {"track record", "edge board"})

    def test_coverage_needs_an_on_topic_claim(self):
        report = gaps.scan(self.index, gaps.open_questions(self.foresea, TODAY), gaps.fund_pairs(self.foresea), TODAY)
        self.assertEqual((report["questions"], report["covered"]), (3, 1))
        self.assertEqual(report["covered_questions"][0]["question"], "Will widget demand pass 110 million units by 2030?")
        unanswered = {g["question"]: g for g in report["gaps"]}
        # Plain agreement would have answered the prime-minister question with Ruritania claims.
        self.assertGreater(unanswered["Will Ruritania's prime minister resign in 2027?"]["off_topic"], 0)
        self.assertEqual(report["by_domain"]["economics"], {"questions": 1, "covered": 1})

    def test_fund_pairs_report_views_and_what_is_missing(self):
        report = gaps.scan(self.index, [], gaps.fund_pairs(self.foresea), TODAY)
        pairs = {p["key"]: p for p in report["fund_pairs"]}
        self.assertNotIn("gold-mcx-kalshi", pairs)  # one commodity on two venues
        self.assertTrue(pairs["copper-aluminium"]["covered"])
        self.assertEqual(pairs["gold-silver"]["missing"],
                         ["no gold price forecast for 2026, 2027", "no silver price forecast for 2026, 2027"])
        self.assertEqual((report["pairs"], report["pairs_covered"]), (2, 1))

    def test_the_report_is_written_and_history_keeps_one_line_a_day(self):
        report = gaps.scan(self.index, gaps.open_questions(self.foresea, TODAY), gaps.fund_pairs(self.foresea), TODAY)
        out = Path(self.tmp.name) / "gaps"
        gaps.write(report, out)
        gaps.write(report, out)
        page = (out / "latest.md").read_text()
        self.assertIn("## Unanswered questions", page)
        self.assertIn("Will a painting sell for $120 million in 2030?", page)
        self.assertIn("gold-silver: no gold price forecast", page)
        self.assertEqual(len((out / "history.jsonl").read_text().splitlines()), 1)
        self.assertEqual(json.loads((out / "latest.json").read_text())["covered"], 1)

    def test_topic_words_skip_filler_and_numbers(self):
        words = gaps.topic_words(["Will copper hit $12,000 by December 2027?", "Copper above $11,000 in 2026?"])
        self.assertEqual(words, [("copper", 2)])

    def test_cli_writes_the_report(self):
        import knowledge_gaps as cli
        out = Path(self.tmp.name) / "report"
        with contextlib.redirect_stdout(io.StringIO()) as printed, \
                mock.patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": ""}):
            code = cli.main(["--kb", str(self.base), "--foresea", str(self.foresea), "--out", str(out),
                             "--today", TODAY.isoformat()])
        self.assertEqual(code, 0)
        self.assertIn("1 of 3 open long-horizon questions covered", printed.getvalue())
        self.assertTrue((out / "latest.md").exists())


if __name__ == "__main__":
    unittest.main()
