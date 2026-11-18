"""Reading reports with the self-hosted model: the live-traffic gate, the gates on claims, review
and publish.

The model, the live site and the report are all fakes with made-up text, so the tests never touch
the network, the model or a real report.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402
from analyzing_llm_rationale import knowledge_reports as reports  # noqa: E402
from test_knowledge import _write, make_base  # noqa: E402
from test_knowledge_watch import PDF, crawler  # noqa: E402

JID = "acme-2026-11-widget-outlook"
PAGES = [
    "Contents\nIntroduction ........ 3\nDemand ........ 5\nSupply ........ 9\nRisks ........ 12",
    "1.1. Demand\nWidget demand in Ruritania is projected to reach 140 million units in 2031, as factories "
    "keep expanding and prices keep falling across the whole region according to our central scenario.",
    "1.2. Supply\nGadget output could fall if the main port closes, which would raise widget costs for buyers "
    "everywhere and slow the build-out of new factories in the coming years by a wide margin.",
    "References\nhttp://a.example http://b.example http://c.example doi:1 doi:2",
]


def reply_for(prompt: str) -> str:
    if "140 million" in prompt:
        return json.dumps({"claims": [
            {"kind": "projection", "text": "Acme sees Ruritanian widget demand at 140 million units by 2031.",
             "entities": ["Widgets", "Ruritania"], "metric": "widget demand", "value": 140,
             "unit": "million units", "geo": "Ruritania", "horizon": 2031},
            {"kind": "projection", "text": "Widget demand reaches 175 million units by 2031.", "entities": ["Widgets"]},
            {"kind": "finding", "text": "Widget demand in Ruritania is projected to reach a record as factories "
                                        "keep expanding.", "entities": ["Widgets"]}],
            "relations": [{"subject": "Ruritania", "predicate": "drives", "object": "Widgets", "claim": 0}]})
    return json.dumps({"claims": [
        {"kind": "risk", "text": "A port closure would cut gadget output and lift widget costs.",
         "entities": ["Gadgets", "Widgets"]}],
        "relations": [{"subject": "Gadgets", "predicate": "feeds", "object": "Widgets", "claim": 0}]})


def fake_pages(path: Path) -> list[str]:
    return list(PAGES)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_base(Path(self.tmp.name))
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def queue(self, **kw):
        pdf = Path(self.tmp.name) / "report.pdf"
        pdf.write_bytes(PDF)
        args = dict(jid=JID, title="Widget Outlook", publisher="Acme Research", published="2026-11-10", path=pdf)
        args.update(kw)
        return reports.queue(self.root, **args)

    def read_all(self, call=None, busy=None, deadline=None):
        job = self.queue()
        reports.prepare(self.root, job, pages=fake_pages)
        return job, reports.read(self.root, job, call or reply_for, busy or (lambda starting: None),
                                 deadline=deadline or self.clock() + 3600, sleep=self.clock.sleep, clock=self.clock)


class LiveGateTests(unittest.TestCase):
    NETSTAT = """Active Internet connections (including servers)
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  127.0.0.1.8090         127.0.0.1.61234        ESTABLISHED
tcp4       0      0  127.0.0.1.61234        127.0.0.1.8090         ESTABLISHED
tcp4       0      0  127.0.0.1.8080         127.0.0.1.61300        ESTABLISHED
tcp4       0      0  127.0.0.1.8090         *.*                    LISTEN
"""

    def test_a_connection_into_the_live_port_is_a_request_in_flight(self):
        self.assertEqual(reports.live_connections(8090, netstat=lambda: self.NETSTAT), 1)
        self.assertEqual(reports.live_connections(8090, netstat=lambda: self.NETSTAT.replace("ESTABLISHED", "TIME_WAIT")), 0)

    def test_the_newest_answered_model_request_in_the_access_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.log"
            rows = [{"ts": 100.0, "status": 200, "request": {"uri": "/v1/chat/completions"}},
                    {"ts": 200.0, "status": 401, "request": {"uri": "/.env"}},
                    {"ts": 150.0, "status": 200, "request": {"uri": "/v1/chat/completions"}}]
            log.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n", encoding="utf-8")
            self.assertEqual(reports.last_live_request(log), 150.0)
            self.assertIsNone(reports.last_live_request(Path(tmp) / "missing.log"))

    def test_the_site_must_be_quiet_before_a_passage_but_only_traffic_stops_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "access.log"
            log.write_text(json.dumps({"ts": 1000.0, "status": 200, "request": {"uri": "/v1/x"}}) + "\n")
            now = [1060.0]
            busy = reports.live_gate(port=1, log=log, idle_s=120, clock=lambda: now[0])
            with unittest.mock.patch.object(reports, "live_connections", return_value=0):
                self.assertIn("60s ago", busy(True))
                self.assertIsNone(busy(False))
                now[0] = 1200.0
                self.assertIsNone(busy(True))
            with unittest.mock.patch.object(reports, "live_connections", return_value=2):
                self.assertIn("in flight", busy(False))


class FakeStream:
    def __init__(self, lines):
        self.lines, self.closed = lines, False

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


class StreamTests(unittest.TestCase):
    def lines(self, *parts):
        return [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}" for p in parts] + ["", "data: [DONE]"]

    def test_a_streamed_answer_is_put_together(self):
        stream = FakeStream(self.lines('{"claims"', ": []}"))
        sent = {}

        def post(url, **kw):
            sent.update(kw["json"])
            return stream
        self.assertEqual(reports.stream_call("p", lambda s: None, post=post, check_s=0), '{"claims": []}')
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["temperature"], 0.0)
        self.assertTrue(stream.closed)

    def test_live_traffic_hangs_up_mid_answer(self):
        stream = FakeStream(self.lines("a", "b", "c"))
        calls = []

        def busy(starting):
            calls.append(starting)
            return "1 live request(s) in flight" if len(calls) >= 2 else None
        with self.assertRaises(reports.Busy):
            reports.stream_call("p", busy, post=lambda url, **kw: stream, check_s=0)
        self.assertTrue(stream.closed)
        self.assertEqual(set(calls), {False})


class PassageTests(unittest.TestCase):
    def test_contents_references_and_tables_are_not_read(self):
        chunks = kb.chunk_pages("x", PAGES)
        kept = [c["page"] for c in chunks if reports.worth_reading(c)]
        self.assertEqual(kept, [2, 3])
        table = {"text": " ".join(str(n) for n in range(60)) + " total"}
        self.assertFalse(reports.worth_reading(table))

    def test_the_report_pdf_is_found_on_its_page(self):
        one = '<a href="/files/full.pdf">Download</a><a href="/about">About</a>'
        self.assertEqual(reports.find_pdf(one, "https://x.org/r/")[0], "https://x.org/files/full.pdf")
        many = ('<a href="/a/summary.pdf">Executive summary</a><a href="/a/WidgetOutlook2026.pdf">Full report</a>'
                '<a href="/a/annex.pdf">Annex</a>')
        self.assertEqual(reports.find_pdf(many, "https://x.org/", "Widget Outlook 2026")[0],
                         "https://x.org/a/WidgetOutlook2026.pdf")
        unclear = '<a href="/a/one.pdf">One</a><a href="/a/two.pdf">Two</a>'
        url, options = reports.find_pdf(unclear, "https://x.org/", "Widget Outlook")
        self.assertIsNone(url)
        self.assertEqual(len(options), 2)

    def test_near_copies_of_a_claim_are_one_candidate(self):
        self.assertTrue(reports.same_claim("Remote workdays could reduce car oil consumption by 2% to 6%.",
                                           "Three more remote workdays could cut car oil consumption by 2% to 6%."))
        self.assertFalse(reports.same_claim("Remote workdays could reduce car oil use by 2% to 6%.",
                                            "Remote workdays could reduce car oil use by 20%."))
        self.assertFalse(reports.same_claim("Speed limits cut oil use by 1% to 6%.",
                                            "Public transport cuts national oil use by 1% to 6%."))

    def test_the_night_window_wraps_midnight(self):
        self.assertTrue(reports.in_window("23-7", datetime(2026, 11, 17, 23, 30)))
        self.assertTrue(reports.in_window("23-7", datetime(2026, 11, 18, 6, 59)))
        self.assertFalse(reports.in_window("23-7", datetime(2026, 11, 18, 7, 0)))
        self.assertTrue(reports.in_window("0-24", datetime(2026, 11, 18, 13, 0)))


class QueueTests(Base):
    def test_bad_ids_dates_files_and_existing_sources_are_refused(self):
        with self.assertRaisesRegex(reports.ReportError, "bad id"):
            self.queue(jid="Acme Outlook")
        with self.assertRaisesRegex(reports.ReportError, "published"):
            self.queue(published="Nov 2026")
        not_pdf = Path(self.tmp.name) / "x.pdf"
        not_pdf.write_bytes(b"<html>")
        with self.assertRaisesRegex(reports.ReportError, "not a PDF"):
            self.queue(path=not_pdf)
        existing = next(p.name for p in (self.root / "sources").iterdir())
        with self.assertRaisesRegex(reports.ReportError, "already in the base"):
            self.queue(jid=existing)

    def test_new_report_editions_in_the_inbox_become_jobs_once(self):
        _write(self.root / "inbox.jsonl", [
            {"found": "2026-11-17", "source": "acme-reports", "publisher": "Acme Research", "title": "Acme reports",
             "kind": "report", "edition": "November 10, 2026", "url": "https://acme.example/r",
             "files": [{"url": "https://acme.example/r/outlook.pdf", "sha256": "abc"}], "status": "new"},
            {"found": "2026-11-17", "source": "acme-tables", "kind": "table", "edition": "x", "status": "new"},
            {"found": "2026-11-17", "source": "acme-reports", "kind": "report", "edition": "old", "status": "read"},
        ])
        queued = reports.from_inbox(self.root)
        self.assertEqual([j["id"] for j in queued], ["acme-2026-11-november-10"])
        job = queued[0]
        self.assertEqual((job["published"], job["file_url"], job["sha256"]),
                         ("2026-11-10", "https://acme.example/r/outlook.pdf", "abc"))
        self.assertEqual(reports.from_inbox(self.root), [])

    def test_a_stuck_job_can_be_queued_again_with_its_file(self):
        job = reports.queue(self.root, jid=JID, title="T", publisher="P", published="2026-11-10",
                            url="https://acme.example/r")
        job.update(status="stuck", error="could not tell which of 3 PDF links is the report", fetch_failures=3)
        reports.save_job(self.root, job)
        again = reports.queue(self.root, jid=JID, title="T", publisher="P", published="2026-11-10",
                              file_url="https://acme.example/r/full.pdf")
        self.assertEqual((again["status"], again["file_url"]), ("new", "https://acme.example/r/full.pdf"))
        self.assertNotIn("error", again)

    def test_a_downloaded_file_must_match_the_inbox_hash(self):
        job = reports.queue(self.root, jid=JID, title="T", publisher="P", published="2026-11-10",
                            file_url="https://acme.example/o.pdf", sha256="0" * 64)
        c, _ = crawler({"https://acme.example/o.pdf": (PDF, "application/pdf")})
        with self.assertRaisesRegex(reports.ReportError, "sha256"):
            reports.fetch_file(self.root, job, c)


class ReadTests(Base):
    def test_gates_keep_grounded_own_words_claims_and_the_rest_waits_for_review(self):
        job, stats = self.read_all()
        self.assertTrue(stats["finished"])
        self.assertEqual((stats["read"], stats["claims"], stats["dropped"]), (2, 2, 2))
        job = reports.load_job(self.root, JID)
        self.assertEqual((job["status"], job["candidates"], job["gate_dropped"]), ("review", 2, 2))
        rows = reports._read_jsonl(reports.folder(self.root, JID) / "candidates.jsonl")
        self.assertEqual([r["n"] for r in rows], ["c001", "c002"])
        self.assertTrue(all(r["review"] is None for r in rows))
        sheet = reports.review_sheet(self.root, JID)
        self.assertIn("c001 projection: Acme sees Ruritanian widget demand", sheet)
        self.assertIn("> 1.1. Demand", sheet)
        self.assertFalse((self.root / "sources" / JID).exists())

    def test_the_model_is_handed_back_whenever_the_site_needs_it(self):
        states = iter(["live request 30s ago", None, None, None, None])
        interrupted = []

        def call(prompt):
            if not interrupted:
                interrupted.append(1)
                raise reports.Busy("1 live request(s) in flight")
            return reply_for(prompt)
        job, stats = self.read_all(call=call, busy=lambda starting: next(states, None))
        self.assertTrue(stats["finished"])
        self.assertEqual((stats["interrupted"], stats["read"], stats["paused_s"]), (1, 2, 30.0))

    def test_reading_stops_at_the_deadline_and_resumes(self):
        def slow(prompt):
            self.clock.now += 100
            return reply_for(prompt)
        job, stats = self.read_all(call=slow, deadline=self.clock() + 50)
        self.assertFalse(stats["finished"])
        self.assertEqual(reports.unread(self.root, job), 1)
        stats = reports.read(self.root, reports.load_job(self.root, JID), reply_for, lambda s: None,
                             deadline=self.clock() + 3600, sleep=self.clock.sleep, clock=self.clock)
        self.assertTrue(stats["finished"])
        self.assertEqual(stats["read"], 1)

    def test_a_passage_that_keeps_failing_is_recorded_and_skipped(self):
        def broken(prompt):
            if "140 million" in prompt:
                raise TimeoutError("model stuck")
            return reply_for(prompt)
        job, stats = self.read_all(call=broken)
        self.assertTrue(stats["finished"])
        job = reports.load_job(self.root, JID)
        self.assertEqual((job["failed_passages"], job["candidates"]), (1, 1))


class PublishTests(Base):
    def setUp(self):
        super().setUp()
        self.read_all()

    def test_every_candidate_needs_a_decision(self):
        with self.assertRaisesRegex(reports.ReportError, "no decision"):
            reports.publish(self.root, JID, reviewer="Claude")
        with self.assertRaisesRegex(reports.ReportError, "no such"):
            reports.decide(self.root, JID, drop=["c009"])

    def test_kept_claims_enter_the_base_with_their_entities_and_relations(self):
        _write(self.root / "inbox.jsonl", [{"source": "acme-reports", "edition": "x", "status": "new"}])
        job = reports.load_job(self.root, JID)
        job["inbox"] = {"source": "acme-reports", "edition": "x"}
        reports.save_job(self.root, job)
        self.assertEqual(reports.decide(self.root, JID, drop=["c002"], rest="keep"),
                         {"keep": 1, "drop": 1, "undecided": 0})
        reports.publish(self.root, JID, reviewer="Claude", source_extra={"licence": "CC BY 4.0"})
        corpus = kb.load(self.root)
        self.assertEqual(corpus.errors, [])
        claims = [c for c in corpus.claims if c["source"] == JID]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["entities"], ["ruritania", "widgets"])
        self.assertEqual(claims[0]["extracted_by"], "self-hosted Qwen3.5-9B, reviewed")
        source = json.loads((self.root / "sources" / JID / "source.json").read_text())
        self.assertEqual((source["licence"], source["kind"]), ("CC BY 4.0", "report"))
        self.assertIn("reviewed by Claude", source["extracted_by"])
        relations = [r for r in corpus.relations if r.get("source", JID) == JID]
        self.assertEqual([(r["subject"], r["object"]) for r in relations if r.get("claims")], [("ruritania", "widgets")])
        chunks = reports._read_jsonl(self.root / "sources" / JID / "chunks.jsonl")
        self.assertEqual([c["page"] for c in chunks], [2])
        self.assertEqual(reports._read_jsonl(self.root / "inbox.jsonl")[0]["status"], "read")
        self.assertEqual(reports.load_job(self.root, JID)["status"], "published")

    def test_only_known_or_mapped_entities_reach_the_graph(self):
        reports.decide(self.root, JID, rest="keep",
                       edits={"c002": {"entities": ["Gadgets", "Widgets", "Port Authority", "governments"]}})
        names = reports.entity_names(self.root, JID)
        self.assertEqual(names["known"]["Gadgets"], "gadgets")
        self.assertEqual(names["new"], {"Port Authority": 1, "governments": 1})
        with self.assertRaisesRegex(reports.ReportError, "neither in the base"):
            reports.publish(self.root, JID, reviewer="Claude", entity_map={"map": {"Port Authority": "ports"}})
        reports.publish(self.root, JID, reviewer="Claude", entity_map={
            "map": {"Port Authority": "ports", "governments": None},
            "new": {"ports": {"name": "Ports", "type": "infrastructure", "aliases": ["port"]}}})
        corpus = kb.load(self.root)
        self.assertEqual(corpus.errors, [])
        self.assertEqual(corpus.entities["ports"]["type"], "infrastructure")
        risk = next(c for c in corpus.claims if c["source"] == JID and c["kind"] == "risk")
        self.assertEqual(risk["entities"], ["gadgets", "ports", "widgets"])
        self.assertNotIn("governments", corpus.entities)
        self.assertEqual(reports.load_job(self.root, JID)["unlinked_entities"], [])

    def test_unmapped_new_names_are_left_off_the_graph(self):
        reports.decide(self.root, JID, rest="keep", edits={"c002": {"entities": ["Gadgets", "costs"]}})
        reports.publish(self.root, JID, reviewer="Claude")
        self.assertNotIn("costs", kb.load(self.root).entities)
        self.assertEqual(reports.load_job(self.root, JID)["unlinked_entities"], ["costs"])

    def test_a_correction_must_still_pass_the_gates(self):
        reports.decide(self.root, JID, rest="keep",
                       edits={"c001": {"text": "Acme sees Ruritanian widget demand at 150 million units by 2031."}})
        with self.assertRaisesRegex(reports.ReportError, "c001: a number"):
            reports.publish(self.root, JID, reviewer="Claude")
        self.assertFalse((self.root / "sources" / JID).exists())
        with self.assertRaisesRegex(reports.ReportError, "cannot edit"):
            reports.decide(self.root, JID, edits={"c001": {"source": "x"}})

    def test_cli_status_and_sheet(self):
        import knowledge_reports as cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["--kb", str(self.root), "status"]), 0)
            self.assertEqual(cli.main(["--kb", str(self.root), "sheet", JID]), 0)
        self.assertIn(f"| {JID} | review |", out.getvalue())
        self.assertIn("# Review", out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["--kb", str(self.root), "publish", JID, "--reviewer", "x"]), 1)
        self.assertIn("no decision", err.getvalue())


if __name__ == "__main__":
    unittest.main()
