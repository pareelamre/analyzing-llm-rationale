"""Table readers: publishers' forecast tables -> knowledge-base claims, with plausibility checks.

Every publisher here is a fake built in memory (a tiny spreadsheet, two web tables, a JSON API),
with made-up numbers, so the tests never touch the network or a real report.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzing_llm_rationale import fund_knowledge  # noqa: E402
from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_tables as tables  # noqa: E402
from test_knowledge import _write  # noqa: E402
from test_knowledge_watch import crawler  # noqa: E402

TODAY = date(2026, 11, 17)


# -- a minimal .xlsx --------------------------------------------------------------------------

def _col(i: int) -> str:
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def xlsx(sheets: dict[str, list[list]]) -> bytes:
    strings: list[str] = []

    def cell(ref, value):
        if value is None:
            return ""
        if isinstance(value, str):
            if value not in strings:
                strings.append(value)
            return f'<c r="{ref}" t="s"><v>{strings.index(value)}</v></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'

    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        entries, rels = [], []
        for n, (name, rows) in enumerate(sheets.items(), 1):
            body = "".join(f'<row r="{r}">' + "".join(cell(f"{_col(c)}{r}", v) for c, v in enumerate(row)) + "</row>"
                           for r, row in enumerate(rows, 1))
            z.writestr(f"xl/worksheets/sheet{n}.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
            entries.append(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>')
            rels.append(f'<Relationship Id="rId{n}" Target="worksheets/sheet{n}.xml"/>')
        z.writestr("xl/workbook.xml", f"<workbook {ns} {rel}><sheets>{''.join(entries)}</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>')
        z.writestr("xl/sharedStrings.xml", f"<sst {ns}>" + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    return buf.getvalue()


MONTHS = [y * 100 + m for y in (2026, 2027) for m in range(1, 13)]


def steo_book(edition="November 2026", brent=80.0, gas=3.0) -> bytes:
    dates = [["Forecast Month -", None, None, edition], ["Last Historical Month--- ", None, None, 202610],
             [None, "AAAA_DATEX or AAAA_YEAR", *MONTHS]]

    def series(code, label, base):
        return [code, label, *[base + (i % 12) for i in range(len(MONTHS))]]
    prices = [[None, "Crude Oil"], series("BREPUUS", "Brent Spot Average", brent),
              series("WTIPUUS", "WTI", brent - 5), series("NGHHUUS", "Henry Hub", gas),
              series("MGRARUS_$", "Gasoline", 3.0), series("DSRTUUS_$", "Diesel", 4.0),
              series("ESRCUUS", "Residential", 15.0)]
    supply = [series("COPRPUS", "US crude", 13.0), series("NGPRPUS", "US gas", 110.0),
              series("papr_world", "World production", 100.0), series("patc_world", "World consumption", 101.0)]
    return xlsx({"Dates": dates, "2tab": prices, "4atab": supply})


def fed_page(funds=("4.1", "3.6", "3.1", "3.0", "3.0")) -> str:
    years = ["2026", "2027", "2028", "2029", "Longer run"]
    rows = [["Variable", "Median 1"], years * 2,
            ["Change in real GDP", "2.3", "2.4", "2.2", "2.1", "2.0", *["x"] * 5],
            ["June projection", "2.2", "2.3", "2.2", "", "2.0", *["x"] * 5],
            ["Unemployment rate", "4.1", "4.1", "4.1", "4.1", "4.2", *["x"] * 5],
            ["PCE inflation", "3.0", "2.3", "2.1", "2.0", "2.0", *["x"] * 5],
            ["Core PCE inflation 4", "3.1", "2.5", "2.2", "2.0", "", *["x"] * 5],
            ["Memo: Projected appropriate policy path"],
            ["Federal funds rate", *funds, *["x"] * 5]]
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<html><table>{body}</table></html>"


def _table(rows) -> str:
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


#: Made-up three-month EURIBOR assumptions: 2025..2028, then revisions for 2026..2028.
EURIBOR = ("1.9", "2.6", "3.4", "3.1", "0.0", "0.3", "-0.2")


def ecb_page(published="2026-09-10", row=("0.9", "1.4", "1.5", "3.0", "2.5", "2.1", "2.5", "2.6", "2.3"),
             euribor=EURIBOR, edition="September 2026") -> str:
    comparison = [["", "Date of release", "GDP growth", "HICP inflation", "HICP inflation excluding energy and food"],
                  ["2026", "2027", "2028"] * 3, ["ECB staff projections", "September 2026", *row],
                  ["European Commission", "May 2026", *["1.0"] * 9]]
    assumptions = [["", edition, "Revisions vs June 2026"], ["2025", "2026", "2027", "2028", "2026", "2027", "2028"],
                   ["Oil price (USD/barrel)", *["70.0"] * 4, *["1.0"] * 3],
                   ["Three-month EURIBOR (percentage per annum)", *euribor]]
    meta = f'<meta property="article:published_time"  content="{published}">' if published else ""
    return (f"<html><head>{meta}</head><p>On 7 September 2026 Eurostat said...</p>"
            f"{_table(assumptions) if euribor else ''}{_table(comparison)}</html>")


#: Made-up deposit facility rate changes, as the ECB Data Portal lists them (dates of changes only).
DFR = [("2022-07-06", "0.4"), ("2024-02-07", "0.65"), ("2025-05-14", "0.4"), ("2026-03-04", "0.9"), ("2026-10-07", "1.15")]


def dfr_csv(changes=DFR, key=tables.ECB_DFR_SERIES) -> tuple[str, str]:
    lines = ["KEY,FREQ,TIME_PERIOD,OBS_VALUE,TITLE"]
    lines += [f'{key},B,{d},{v},"Deposit facility - date of changes (raw data) - Level"' for d, v in changes]
    return "\n".join(lines) + "\n", "text/csv"


def imf_api(edition="October 2026") -> dict:
    indicators = {"indicators": {"NGDP_RPCH": {"source": f"World Economic Outlook ({edition})"}}}
    econs = [code for code, *_ in tables.IMF_ECONOMIES]

    def values(code, base):
        return {"values": {code: {e: {str(y): base + i * 0.1 for y in range(2024, 2031)} for i, e in enumerate(econs)}}}
    api = tables.IMF_API
    return {f"{api}/indicators": json.dumps(indicators), f"{api}/NGDP_RPCH": json.dumps(values("NGDP_RPCH", 2.0)),
            f"{api}/PCPIPCH": json.dumps(values("PCPIPCH", 3.0)), f"{api}/LUR": json.dumps(values("LUR", 5.0))}


FED_URL = "https://www.federalreserve.gov/monetarypolicy/fomcprojtabl20260916.htm"
ECB_URL = "https://www.ecb.europa.eu/press/projections/html/ecb.projections202609_ecbstaff~abc.en.html"
STATE = {
    "eia-steo": {"latest": "November 10, 2026", "seen": {"November 10, 2026": {"url": "https://www.eia.gov/outlooks/steo/"}}},
    "fed-sep": {"latest": "20260916", "seen": {"20260916": {"url": FED_URL}}},
    "ecb-projections": {"latest": "202609", "seen": {"202609": {"url": ECB_URL}}},
    "imf-weo": {"latest": "2026/07/08", "seen": {"2026/07/08": {"url": "x"}}},
}


def web(**over) -> dict:
    pages = {tables.STEO_XLSX: (steo_book(), "application/vnd.openxmlformats"), FED_URL: fed_page(), ECB_URL: ecb_page(),
             tables.ECB_DFR_API: dfr_csv(), **imf_api()}
    pages.update(over)
    return pages


def make_root(tmp: str) -> Path:
    root = Path(tmp)
    _write(root / "entities.jsonl", [{"id": eid, "name": eid, "type": t} for eid, t in (
        ("eia", "organization"), ("imf", "organization"), ("crude_oil", "commodity"), ("natural_gas", "commodity"),
        ("electricity_prices", "metric"), ("gdp_growth", "metric"), ("inflation", "metric"), ("us", "country"),
        ("china", "country"), ("india", "country"), ("euro_area", "region"), ("japan", "country"), ("uk", "country"),
        ("germany", "country"), ("france", "country"), ("russia", "country"), ("brazil", "country"),
        ("saudi_arabia", "country"), ("indonesia", "country"))])
    (root / "sources").mkdir()
    (root / "watch_state.json").write_text(json.dumps(STATE), encoding="utf-8")
    return root


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)
        self.corpus = kb.load(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _read(self, rid, pages=None):
        c, _ = crawler(pages or web())
        reader = tables.READERS[rid]
        sid, info = reader.ident(STATE.get(rid, {}), c, self.corpus)
        return sid, reader.read(info, c, TODAY, self.corpus)

    def test_the_spreadsheet_reader_needs_no_library(self):
        book = tables.Workbook(xlsx({"S": [["a", 1.5, None, "b"]]}))
        self.assertEqual(book.rows("S"), [["a", 1.5, None, "b"]])
        with self.assertRaises(tables.TableError):
            tables.Workbook(b"not a zip")

    def test_steo_gives_yearly_averages_and_december_values(self):
        sid, ed = self._read("eia-steo")
        self.assertEqual(sid, "eia-2026-11-steo")
        self.assertEqual(ed.source["published"], "2026-11-10")
        by = {(c["metric"], c["horizon"]): c for c in ed.claims}
        brent27 = by[("Brent crude price", "2027")]
        self.assertEqual(brent27["value"], 85.5)  # mean of 80..91
        self.assertEqual((brent27["unit"], brent27["geo"], brent27["scenario"]), ("$/bbl", "global", "EIA STEO baseline"))
        self.assertIn("$85.5 a barrel in 2027", brent27["text"])
        self.assertIn("actual data through October 2026", by[("Brent crude price", "2026")]["text"])
        self.assertEqual(by[("Brent crude price (December average)", "2027-12")]["value"], 91.0)
        self.assertNotIn(("US retail diesel price (December average)", "2027-12"), by)
        self.assertEqual(len(ed.claims), 28)

    def test_steo_refuses_a_spreadsheet_from_another_month(self):
        c, _ = crawler(web(**{tables.STEO_XLSX: (steo_book("October 2026"), "x")}))
        sid, info = tables.READERS["eia-steo"].ident(STATE["eia-steo"], c, self.corpus)
        with self.assertRaisesRegex(tables.TableError, "October 2026 edition"):
            tables.READERS["eia-steo"].read(info, c, TODAY, self.corpus)

    def test_fed_medians_with_the_previous_projection_and_longer_run(self):
        sid, ed = self._read("fed-sep")
        self.assertEqual((sid, ed.source["published"]), ("fed-2026-09-sep", "2026-09-16"))
        gdp = next(c for c in ed.claims if c["metric"] == "real GDP growth (Q4/Q4)" and c["horizon"] == "2026")
        self.assertIn("2.3% in 2026, fourth quarter over fourth quarter (June projection: 2.2%)", gdp["text"])
        longer = next(c for c in ed.claims if c["metric"] == "federal funds rate (longer run)")
        self.assertEqual((longer["kind"], longer.get("horizon")), ("estimate", None))
        self.assertTrue(any(c["metric"] == "core PCE inflation (Q4/Q4)" for c in ed.claims))
        self.assertEqual(len(ed.claims), 24)

    def test_ecb_reads_its_own_row_and_the_page_date(self):
        sid, ed = self._read("ecb-projections")
        self.assertEqual((sid, ed.source["published"]), ("ecb-2026-09-projections", "2026-09-10"))
        macro = [c for c in ed.claims if c["section"] == "Comparison with other forecasts"]
        self.assertEqual([c["value"] for c in macro], [0.9, 1.4, 1.5, 3.0, 2.5, 2.1, 2.5, 2.6, 2.3])
        self.assertEqual({c["geo"] for c in ed.claims}, {"euro area"})
        _, undated = self._read("ecb-projections", web(**{ECB_URL: ecb_page(published=None)}))
        self.assertEqual(undated.source["published"], "2026-09-30")  # never early

    def test_ecb_reads_the_euribor_path_from_the_projection_year_on(self):
        _, ed = self._read("ecb-projections")
        path = [c for c in ed.claims if c["metric"].startswith("three-month EURIBOR")]
        self.assertEqual([(c["horizon"], c["value"]) for c in path], [("2026", 2.6), ("2027", 3.4), ("2028", 3.1)])
        self.assertEqual({c["kind"] for c in path}, {"projection"})
        self.assertTrue(all("ecb_deposit_rate" in c["entities"] for c in path))
        self.assertIn("unchanged from the June 2026 projections", path[0]["text"])
        self.assertIn("averaging 3.4% in 2027, 0.3 points higher than in the June 2026", path[1]["text"])
        self.assertIn("0.2 points lower", path[2]["text"])
        self.assertEqual([c["id"] for c in ed.claims], [f"ecb-2026-09-projections:c{n:03d}" for n in range(1, 13)])

    def test_ecb_refuses_a_changed_table(self):
        with self.assertRaisesRegex(tables.TableError, "changed shape"):
            self._read("ecb-projections", web(**{ECB_URL: ecb_page(row=("0.9", "1.4"))}))
        with self.assertRaisesRegex(tables.TableError, "no comparison table"):
            self._read("ecb-projections", web(**{ECB_URL: "<html><table><tr><td>x</td></tr></table></html>"}))
        with self.assertRaisesRegex(tables.TableError, "no technical assumptions table"):
            self._read("ecb-projections", web(**{ECB_URL: ecb_page(euribor=())}))
        with self.assertRaisesRegex(tables.TableError, "EURIBOR row changed shape"):
            self._read("ecb-projections", web(**{ECB_URL: ecb_page(euribor=EURIBOR[:5])}))
        with self.assertRaisesRegex(tables.TableError, "not the September 2026 edition"):
            self._read("ecb-projections", web(**{ECB_URL: ecb_page(edition="June 2026")}))
        with self.assertRaisesRegex(tables.TableError, "outside"):
            self._read("ecb-projections", web(**{ECB_URL: ecb_page(euribor=("1.9", "2.6", "34", "3.1", "0", "0", "0"))}))

    def test_ecb_rates_are_the_recent_changes_dated_when_they_took_effect(self):
        sid, ed = self._read("ecb-rates")
        self.assertEqual((sid, ed.source["published"]), ("ecb-2026-10-deposit-rate", "2026-10-07"))
        self.assertEqual([(c["horizon"], c["value"]) for c in ed.claims],
                         [("2024-02-07", 0.65), ("2025-05-14", 0.4), ("2026-03-04", 0.9), ("2026-10-07", 1.15)])
        self.assertEqual({(c["kind"], c["unit"], c["metric"]) for c in ed.claims}, {("fact", "%", "ECB deposit facility rate")})
        self.assertEqual(ed.claims[1]["text"], "The ECB cut its deposit facility rate, the rate it steers policy with, "
                                               "by 25 basis points, from 0.65% to 0.40%, effective 14 May 2025.")
        self.assertIn("raised its deposit facility rate, the rate it steers policy with, by 50 basis points, "
                      "from 0.40% to 0.90%", ed.claims[2]["text"])

    def test_ecb_rates_refuse_an_odd_series(self):
        for changes, key, why in ((DFR + [("2026-12-02", "3.15")], tables.ECB_DFR_SERIES, "more than"),
                                  (DFR, "FM.B.U2.EUR.4F.KR.MRR_FR.LEV", "unexpected row"),
                                  ([("2026-01-07", "x")], tables.ECB_DFR_SERIES, "unexpected row"),
                                  ([("2026-01-07", "40")], tables.ECB_DFR_SERIES, "outside"),
                                  ([], tables.ECB_DFR_SERIES, "no deposit facility rate changes")):
            with self.subTest(why=why), self.assertRaisesRegex(tables.TableError, why):
                self._read("ecb-rates", web(**{tables.ECB_DFR_API: dfr_csv(changes, key)}))

    def test_imf_reads_the_edition_from_the_api_and_notes_india_fiscal_years(self):
        sid, ed = self._read("imf-weo")
        self.assertEqual((sid, ed.source["published"]), ("imf-2026-10-weo-data", "2026-10-31"))
        india = [c for c in ed.claims if c["geo"] == "India"]
        self.assertTrue(all("fiscal year" in c["text"] for c in india if c["metric"] != "unemployment rate"))
        self.assertFalse(any("fiscal year" in c["text"] for c in india if c["metric"] == "unemployment rate"))
        self.assertEqual({c["horizon"] for c in ed.claims}, {"2026", "2027", "2028"})
        self.assertEqual(len(ed.claims), 13 * 3 * 3)


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_out_of_range_and_too_few_values_are_refused(self):
        claim = tables._claim("x", 1, text="t", metric="Brent crude price", value=4000, unit="$/bbl", geo="global",
                              horizon="2027", scenario=tables.STEO_SCENARIO, entities=[], section="s")
        problems = tables.check([claim], kb.load(self.root), tables.STEO_SCENARIO, 16, 1)
        self.assertTrue(any("outside" in p for p in problems))
        self.assertTrue(any("only 1 values" in p for p in problems))

    def test_a_big_move_from_the_last_edition_needs_a_person(self):
        c, _ = crawler(web())
        self.assertEqual(tables.update(self.root, c, today=TODAY, only=["eia-steo"])[0]["status"], "added")
        state = json.loads((self.root / "watch_state.json").read_text())
        state["eia-steo"]["latest"] = "December 9, 2026"
        (self.root / "watch_state.json").write_text(json.dumps(state))
        c, _ = crawler(web(**{tables.STEO_XLSX: (steo_book("December 2026", brent=200.0), "x")}))
        result = tables.update(self.root, c, today=TODAY, only=["eia-steo"])[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("moved from", result["detail"])
        self.assertFalse((self.root / "sources" / "eia-2026-12-steo").exists())

    def test_duplicates_of_the_same_publisher_are_dropped_and_ids_renumbered(self):
        folder = self.root / "sources" / "imf-2026-10-chapter"
        _write(folder / "source.json", {"id": "imf-2026-10-chapter", "title": "WEO", "publisher": "IMF",
                                        "published": "2026-10-13"})
        _write(folder / "claims.jsonl", [{"id": "imf-2026-10-chapter:c001", "kind": "projection", "text": "US growth",
                                          "metric": "real GDP growth", "value": 2.0, "unit": "%", "geo": "US",
                                          "horizon": "2026", "entities": ["us"]}])
        corpus = kb.load(self.root)
        mk = lambda v, h: tables._claim("new", 1, text="t", metric="real GDP growth", value=v, unit="%", geo="US",  # noqa: E731
                                        horizon=h, scenario="s", entities=[], section="s")
        kept = tables.dedupe([mk(2.0, "2026"), mk(2.1, "2027")], corpus, "IMF")
        self.assertEqual([(c["horizon"], c["id"]) for c in kept], [("2027", "new:c001")])
        self.assertEqual(len(tables.dedupe([mk(2.0, "2026")], corpus, "World Bank")), 1)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_reader_adds_its_edition_once_and_the_base_loads(self):
        _write(self.root / "inbox.jsonl", [{"source": "fed-sep", "edition": "20260916", "status": "new"}])
        c, _ = crawler(web())
        results = tables.update(self.root, c, today=TODAY)
        self.assertEqual({r["reader"]: r["status"] for r in results},
                         {"eia-steo": "added", "fed-sep": "added", "ecb-projections": "added", "ecb-rates": "added",
                          "imf-weo": "added"})
        corpus = kb.load(self.root)
        self.assertEqual(corpus.errors, [])
        self.assertIn("federal_funds_rate", corpus.entities)
        self.assertIn("ecb_deposit_rate", corpus.entities)
        inbox = json.loads((self.root / "inbox.jsonl").read_text().splitlines()[0])
        self.assertEqual((inbox["status"], inbox["kb_source"]), ("read", "fed-2026-09-sep"))
        c, _ = crawler(web())
        self.assertEqual({r["status"] for r in tables.update(self.root, c, today=TODAY)}, {"current"})

    def test_a_dry_run_writes_nothing_and_a_failed_reader_does_not_stop_the_rest(self):
        c, _ = crawler(web(**{FED_URL: 500}))
        results = {r["reader"]: r for r in tables.update(self.root, c, today=TODAY, dry_run=True)}
        self.assertEqual(results["fed-sep"]["status"], "failed")
        self.assertEqual(results["eia-steo"]["status"], "would add")
        self.assertEqual(list((self.root / "sources").iterdir()), [])

    def test_a_write_that_breaks_the_base_is_undone(self):
        before = (self.root / "entities.jsonl").read_text()
        c, _ = crawler(web())
        reader = tables.READERS["fed-sep"]
        corpus = kb.load(self.root)
        edition = reader.read(reader.ident(STATE["fed-sep"], c, corpus)[1], c, TODAY, corpus)
        broken = kb.Corpus(root=self.root)
        broken.errors.append("boom")
        with mock.patch.object(tables.knowledge, "load", return_value=broken):
            with self.assertRaisesRegex(tables.TableError, "would not load"):
                tables.write(self.root, edition)
        self.assertFalse((self.root / "sources" / "fed-2026-09-sep").exists())
        self.assertEqual((self.root / "entities.jsonl").read_text(), before)

    def test_a_new_rate_decision_adds_only_itself(self):
        c, _ = crawler(web())
        tables.update(self.root, c, today=TODAY, only=["ecb-rates"])
        c, _ = crawler(web(**{tables.ECB_DFR_API: dfr_csv(DFR + [("2026-12-09", "1.4")])}))
        result = tables.update(self.root, c, today=TODAY, only=["ecb-rates"])[0]
        self.assertEqual((result["status"], result["source"], result["claims"]), ("added", "ecb-2026-12-deposit-rate", 1))
        c, _ = crawler(web(**{tables.ECB_DFR_API: dfr_csv(DFR + [("2026-12-09", "1.4")])}))
        self.assertEqual(tables.update(self.root, c, today=TODAY, only=["ecb-rates"])[0]["status"], "current")

    def test_an_ecb_rate_question_leads_with_the_ecb_rate_the_asked_year_first(self):
        c, _ = crawler(web())
        tables.update(self.root, c, today=TODAY)
        index = kb.Index(kb.load(self.root))
        self.assertEqual(index.topics("Will the ECB cut rates in 2027?"), ["ecb", "ecb_deposit_rate"])
        hits = index.agreed("Will the ECB cut rates in 2027?", as_of=TODAY)
        self.assertTrue(hits)
        first = hits[0].claim
        self.assertEqual((first["metric"], first["horizon"]), ("three-month EURIBOR (market-based assumption)", "2027"))
        lead = [h.claim for h in hits if "ecb_deposit_rate" in h.claim["entities"]]
        self.assertEqual([h.claim for h in hits[:len(lead)]], lead)  # the rate's claims come first
        # Next, a year away: the 2026 decisions, newest first, then the 2026 and 2028 assumptions.
        self.assertEqual([c["id"] for c in lead[1:5]],
                         ["ecb-2026-10-deposit-rate:c004", "ecb-2026-10-deposit-rate:c003",
                          "ecb-2026-09-projections:c010", "ecb-2026-09-projections:c012"])
        # With no year asked, the latest decision leads.
        now = index.agreed("Will the ECB cut rates at its next meeting?", as_of=TODAY)
        self.assertEqual([h.claim["id"] for h in now[:2]], ["ecb-2026-10-deposit-rate:c004", "ecb-2026-10-deposit-rate:c003"])
        for hit in hits:
            self.assertFalse({"federal_funds_rate", "federal_reserve"} & set(hit.claim["entities"]), hit.claim["text"])
        # A Fed question gets the Fed's path, never the ECB's.
        fed = index.agreed("Will the Fed cut rates in 2027?", as_of=TODAY)
        self.assertTrue(fed and "federal_funds_rate" in fed[0].claim["entities"])
        self.assertFalse(any("ecb_deposit_rate" in h.claim["entities"] for h in fed))

    def test_the_fund_reads_the_newest_energy_forecast(self):
        c, _ = crawler(web())
        tables.update(self.root, c, today=TODAY, only=["eia-steo"])
        view = fund_knowledge.pair_view(kb.Index(kb.load(self.root)), "crude", "natgas", TODAY)
        self.assertEqual(view["source"], "eia-2026-11-steo")
        self.assertEqual(view["years"], [2026, 2027])

    def test_cli_dry_run_prints_the_table(self):
        import knowledge_tables as cli
        c, _ = crawler(web())
        with mock.patch.object(cli.watch, "Crawler", return_value=c), \
                mock.patch.dict("os.environ", {"GITHUB_ACTIONS": "", "GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--kb", str(self.root), "--dry-run"]), 0)
        self.assertIn("| eia-steo | would add | eia-2026-11-steo | 28 |", out.getvalue())


if __name__ == "__main__":
    unittest.main()
