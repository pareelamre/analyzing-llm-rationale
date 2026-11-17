"""Exchange readers: CFTC positioning on COMEX and NYMEX, Shanghai futures settlements.

The CFTC API and the exchange's data file are fakes with made-up numbers, so the tests never touch
the network.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_markets as markets  # noqa: E402
from analyzing_llm_rationale import knowledge_tables as tables  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402
from test_knowledge import _write  # noqa: E402

TODAY = date(2026, 11, 17)
WEEKS = ["2026-11-10", "2026-11-03", "2026-10-27", "2026-10-20", "2026-10-13", "2026-10-06"]


def cot_rows(gold_long=(130000, 120000, 118000, 115000, 110000, 100000)) -> list[dict]:
    rows = []
    for i, week in enumerate(WEEKS):
        for code in markets.COT_MARKETS:
            long_ = gold_long[i] if code == "088691" else 20000 + i * 100
            rows.append({"cftc_contract_market_code": code, "report_date_as_yyyy_mm_dd": f"{week}T00:00:00.000",
                         "open_interest_all": "500000", "m_money_positions_long_all": str(long_),
                         "m_money_positions_short_all": "15000"})
    return rows


def shfe_rows(copper=105040) -> dict:
    rows = []
    for product, months in (("cu_f", ["2611", "2612", "2701", "2703", "2706", "2709"]),
                            ("al_f", ["2612", "2703", "2706", "2709"]), ("zn_f", ["2612", "2703", "2706"]),
                            ("ni_f", ["2612", "2703", "2706"]), ("au_f", ["2612", "2702", "2706"]),
                            ("ag_f", ["2612", "2702", "2706"])):
        base = {"cu_f": copper, "al_f": 22660, "zn_f": 25190, "ni_f": 122410, "au_f": 838.2, "ag_f": 12032}[product]
        for n, month in enumerate(months):
            rows.append({"PRODUCTID": product + "  ", "DELIVERYMONTH": month, "SETTLEMENTPRICE": base - n,
                         "OPENINTEREST": 100000 if month == "2701" or (product != "cu_f" and n == 0) else 5000 - n})
        rows.append({"PRODUCTID": product, "DELIVERYMONTH": "小计", "SETTLEMENTPRICE": "", "OPENINTEREST": 1})
    rows.append({"PRODUCTID": "rb_f", "DELIVERYMONTH": "2701", "SETTLEMENTPRICE": 3100, "OPENINTEREST": 9})
    return {"o_curinstrument": rows}


class FakeExchange:
    def __init__(self, cot=None, shfe=None):
        self.cot = cot_rows() if cot is None else cot
        self.shfe = shfe or {"20261113": shfe_rows()}
        self.calls = []

    def fetch(self, url, *, max_bytes=watch.PAGE_BYTES):
        self.calls.append(url)
        if url.startswith(markets.COT_API):
            limit = int(url.split("%24limit=")[1].split("&")[0])
            return watch.Response(200, json.dumps(self.cot[:limit]).encode(), "application/json", url)
        for day, body in self.shfe.items():
            if url == markets.SHFE_DATA.format(day=day):
                if isinstance(body, int):
                    return watch.Response(body, b"", "", url)
                return watch.Response(200, json.dumps(body).encode(), "application/json", url)
        return watch.Response(404, b"", "", url)


def make_root(tmp: str) -> Path:
    root = Path(tmp)
    _write(root / "entities.jsonl", [{"id": e, "name": e, "type": "commodity"} for e in (
        "gold", "silver", "copper", "platinum", "crude_oil", "natural_gas", "gasoline", "aluminium", "zinc", "nickel")]
        + [{"id": "china", "name": "China", "type": "country"}])
    (root / "sources").mkdir()
    return root


class CotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)
        self.corpus = kb.load(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, web):
        reader = markets.READERS["cftc-cot"]
        sid, info = reader.ident({}, web, self.corpus)
        return sid, reader.read(info, web, TODAY, self.corpus)

    def test_net_positions_with_their_share_and_weekly_changes(self):
        sid, edition = self.read(FakeExchange())
        self.assertEqual(sid, "cftc-2026-11-10-cot")
        self.assertEqual(edition.source["published"], "2026-11-13")
        self.assertEqual(len(edition.claims), len(markets.COT_MARKETS))
        gold = edition.claims[0]
        self.assertEqual((gold["metric"], gold["value"], gold["unit"], gold["kind"]),
                         ("COMEX gold money-manager net position", 115000, "contracts", "fact"))
        self.assertEqual(gold["text"], "In the week to 10 November 2026, money managers held a net long position of "
                                       "115,000 COMEX gold futures contracts (100 troy ounces each), 23.0% of open "
                                       "interest; that is up 10,000 from a week earlier and up 20,000 from four weeks "
                                       "earlier.")
        self.assertIn("comex", gold["entities"])
        self.assertTrue(kb.grounded(gold, gold["text"]))

    def test_a_net_short_is_said_as_such(self):
        _, edition = self.read(FakeExchange(cot_rows(gold_long=(5000,) * 6)))
        self.assertIn("a net short position of 10,000 COMEX gold", edition.claims[0]["text"])
        self.assertIn("unchanged from a week earlier", edition.claims[0]["text"])

    def test_positions_larger_than_open_interest_are_refused(self):
        rows = cot_rows(gold_long=(900000,) * 6)
        with self.assertRaisesRegex(tables.TableError, "exceeds open interest"):
            self.read(FakeExchange(rows))


class ShfeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)
        self.corpus = kb.load(self.root)
        self.today = mock.patch.object(markets, "TODAY", return_value=TODAY)
        self.today.start()

    def tearDown(self):
        self.today.stop()
        self.tmp.cleanup()

    def read(self, web):
        reader = markets.READERS["shfe-futures"]
        sid, info = reader.ident({}, web, self.corpus)
        return sid, reader.read(info, web, TODAY, self.corpus)

    def test_most_traded_and_quarterly_contracts_of_six_metals(self):
        sid, edition = self.read(FakeExchange())
        self.assertEqual(sid, "shfe-2026-11-13-futures")
        copper = [c for c in edition.claims if c["metric"] == "SHFE copper futures price"]
        self.assertEqual([c["horizon"] for c in copper], ["2026-12", "2027-01", "2027-03", "2027-06", "2027-09"])
        active = next(c for c in copper if c["horizon"] == "2027-01")
        self.assertIn("the most traded contract", active["text"])
        self.assertEqual((active["kind"], active["unit"], active["geo"]), ("estimate", "CNY/t", "China"))
        gold = next(c for c in edition.claims if c["metric"] == "SHFE gold futures price")
        self.assertIn("838.2 yuan a gram", gold["text"])
        self.assertTrue(all(kb.grounded(c, c["text"]) for c in edition.claims))
        self.assertFalse(any("rb" in c["section"] for c in edition.claims))

    def test_a_friday_holiday_falls_back_to_thursday(self):
        sid, _ = self.read(FakeExchange(shfe={"20261112": shfe_rows()}))
        self.assertEqual(sid, "shfe-2026-11-12-futures")

    def test_a_server_error_is_reported(self):
        with self.assertRaisesRegex(tables.TableError, "HTTP 500"):
            self.read(FakeExchange(shfe={"20261113": 500}))


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_root(self.tmp.name)
        self.today = mock.patch.object(markets, "TODAY", return_value=TODAY)
        self.today.start()

    def tearDown(self):
        self.today.stop()
        self.tmp.cleanup()

    def test_weekly_editions_supersede_and_old_ones_are_pruned(self):
        web = FakeExchange()
        results = tables.update(self.root, web, today=TODAY, readers=markets.READERS)
        self.assertEqual({r["reader"]: r["status"] for r in results}, {"cftc-cot": "added", "shfe-futures": "added"})
        corpus = kb.load(self.root)
        self.assertEqual(corpus.errors, [])
        self.assertIn("comex", corpus.entities)
        self.assertIn("shfe", corpus.entities)
        self.assertEqual({r["status"] for r in tables.update(self.root, web, today=TODAY, readers=markets.READERS)},
                         {"current"})
        # A week later: positioning keeps one edition, futures keep both and the newer price leads.
        later = FakeExchange(cot=[{**r, "report_date_as_yyyy_mm_dd": "2026-11-17T00:00:00.000"} for r in cot_rows()[:9]]
                             + cot_rows(), shfe={"20261120": shfe_rows(copper=106000)})
        with mock.patch.object(markets, "TODAY", return_value=date(2026, 11, 24)):
            results = {r["reader"]: r for r in tables.update(self.root, later, today=TODAY, readers=markets.READERS)}
        self.assertEqual(results["cftc-cot"]["removed"], ["cftc-2026-11-10-cot"])
        self.assertNotIn("removed", results["shfe-futures"])
        names = sorted(p.name for p in (self.root / "sources").iterdir())
        self.assertEqual(names, ["cftc-2026-11-17-cot", "shfe-2026-11-13-futures", "shfe-2026-11-20-futures"])
        index = kb.Index(kb.load(self.root))
        rows = [r for r in index.timeline("SHFE copper futures price December 2026") if r["horizon"] == "2026-12"]
        self.assertEqual(rows[0]["value"], 105999)  # the December contract is the second listed
        self.assertEqual([h["value"] for h in rows[0]["history"]], [105039])

    def test_the_cot_change_counts_contracts_not_percent(self):
        self.assertEqual(markets._signed(-2004), "down 2,004")
        self.assertEqual(markets._signed(0), "unchanged")


if __name__ == "__main__":
    unittest.main()
