"""Foresea's knowledge base: loading, hybrid retrieval, horizons, ingest and the forecast hook.

The fixture is made up (widgets in Ruritania), so no licensed report ever enters this public repo.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyzing_llm_rationale import knowledge as kb  # noqa: E402

OLD, NEW = "acme-2025-01-widgets", "acme-2026-06-widgets"


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        path.write_text(json.dumps(rows), encoding="utf-8")
    else:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_base(root: Path) -> Path:
    _write(root / "entities.jsonl", [
        {"id": "acme", "name": "Acme Research", "type": "organization", "aliases": ["Acme"]},
        {"id": "widgets", "name": "Widgets", "type": "commodity", "aliases": ["widget", "widget demand"]},
        {"id": "gadgets", "name": "Gadgets", "type": "commodity", "aliases": ["gadget"]},
        {"id": "ruritania", "name": "Ruritania", "type": "country", "aliases": ["Ruritanian"]},
    ])
    for sid, published, value in ((OLD, "2025-01-15", 100), (NEW, "2026-06-01", 120)):
        folder = root / "sources" / sid
        _write(folder / "source.json", {"id": sid, "title": "Annual Outlook", "publisher": "Acme Research",
                                        "published": published, "summary": "Widgets."})
        claims = [{"id": f"{sid}:c001", "kind": "projection", "page": 3,
                   "text": f"Acme expects global widget demand to reach {value} million units in 2030.",
                   "entities": ["widgets", "acme"], "metric": "widget demand", "value": value,
                   "unit": "million units", "geo": "global", "horizon": "2030"}]
        if sid == OLD:
            claims += [
                {"id": f"{sid}:c002", "kind": "fact", "page": 5, "text": "Ruritania built three widget factories in 2024.",
                 "entities": ["ruritania", "widgets"]},
                {"id": f"{sid}:c003", "kind": "finding", "page": 7,
                 "text": "Gadget prices fell sharply as factories scaled up.", "entities": ["gadgets"]},
            ]
        else:
            claims.append({"id": f"{sid}:c002", "kind": "policy", "page": 9, "horizon": "2027",
                           "text": "Ruritania will tax gadget imports from 2027.", "entities": ["ruritania", "gadgets"]})
        _write(folder / "claims.jsonl", claims)
        rels = [{"subject": "widgets", "predicate": "substitutes", "object": "gadgets",
                 "claims": [f"{OLD}:c003"]}] if sid == OLD else []
        _write(folder / "relations.jsonl", rels)
        _write(folder / "chunks.jsonl", [{"id": f"{sid}:p3:1", "page": 3, "section": "2. Demand",
                                          "text": f"Widget demand reaches {value} million units in 2030."}])
    return root


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_base(Path(self.tmp.name))
        self.corpus = kb.load(self.root)
        self.index = kb.Index(self.corpus)

    def tearDown(self):
        self.tmp.cleanup()


class LoadingTests(BaseCase):
    def test_a_clean_base_loads_without_errors(self):
        self.assertEqual(self.corpus.errors, [])
        self.assertEqual(kb.stats(self.corpus)["claims"], 5)
        self.assertEqual(kb.stats(self.corpus)["horizons"], [2027, 2030])

    def test_every_broken_reference_is_reported_not_raised(self):
        folder = self.root / "sources" / NEW
        with (folder / "claims.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": f"{NEW}:c001", "kind": "guess", "text": "x", "entities": ["nobody"],
                                 "horizon": "soon"}) + "\nnot json\n")
        _write(folder / "relations.jsonl", [{"subject": "widgets", "predicate": "", "object": "ghost",
                                              "claims": ["missing:c9"]}])
        _write(self.root / "sources" / "bad" / "source.json", {"id": "other", "title": "T", "publisher": "P",
                                                                "published": "Nov 2026"})
        errors = "\n".join(kb.load(self.root).errors)
        for expected in ("duplicate id", "kind must be one of", "unknown entity nobody", "horizon must be",
                         "not JSON", "unknown object 'ghost'", "no predicate", "unknown claim missing:c9",
                         "does not match its folder", "published must be YYYY-MM-DD"):
            self.assertIn(expected, errors)

    def test_dates_cover_years_months_and_days(self):
        self.assertEqual(kb.parse_when("2035"), date(2035, 1, 1))
        self.assertEqual(kb.parse_when("2026-09"), date(2026, 9, 1))
        self.assertIsNone(kb.parse_when("2026-13"))


class RetrievalTests(BaseCase):
    def test_names_are_linked_by_alias_and_longest_first(self):
        self.assertEqual(self.index.link("What will Ruritanian widget demand be?"), ["widgets", "ruritania"])

    def test_a_horizon_question_fuses_text_graph_and_year(self):
        hits = self.index.search("How many widgets will be sold by 2030?", k=5)
        self.assertEqual(hits[0].claim["id"], f"{NEW}:c001")
        self.assertIn("horizon", hits[0].lists)
        self.assertIn("graph", hits[0].lists)
        self.assertEqual(hits[0].cite(), "Acme Research, Jun 2026, p. 3")
        self.assertFalse(hits[0].superseded)

    def test_a_newer_projection_pushes_the_older_one_back(self):
        hits = {h.claim["id"]: h for h in self.index.search("widget demand 2030", k=5)}
        self.assertTrue(hits[f"{OLD}:c001"].superseded)
        self.assertLess(hits[f"{OLD}:c001"].score, hits[f"{NEW}:c001"].score)
        # Before the newer report existed, the older one was the current view.
        early = self.index.search("widget demand 2030", as_of=date(2025, 12, 31), k=1)
        self.assertFalse(early[0].superseded)

    def test_the_graph_reaches_a_neighbour_the_question_never_names(self):
        hits = self.index.search("widget market", k=10)
        gadget = [h for h in hits if h.claim["id"] == f"{OLD}:c003"]
        self.assertTrue(gadget)
        self.assertEqual(gadget[0].lists, ["graph"])

    def test_as_of_hides_reports_published_later(self):
        hits = self.index.search("widget demand 2030", as_of=date(2025, 12, 31))
        self.assertTrue(hits)
        self.assertTrue(all(h.claim["source"] == OLD for h in hits))
        self.assertEqual(self.index.search("widget demand", as_of=date(2024, 1, 1)), [])

    def test_kinds_filter(self):
        hits = self.index.search("Ruritania", kinds=["policy"])
        self.assertEqual([h.claim["id"] for h in hits], [f"{NEW}:c002"])

    def test_a_reranker_reorders_the_head(self):
        def prefer_gadgets(question, texts):
            return [1.0 if "Gadget" in t or "gadget" in t else 0.0 for t in texts]
        hits = self.index.search("Ruritania widget factories", k=3, rerank=prefer_gadgets)
        self.assertIn("gadget", hits[0].claim["text"].lower())

    def test_dense_vectors_join_the_fusion_and_are_cached(self):
        calls = []

        def embed(texts):
            calls.extend(texts)
            return [[1.0, 0.0] if "widget" in t.lower() else [0.0, 1.0] for t in texts]
        cache = self.root / ".kb" / "vectors.sqlite"
        index = kb.Index(self.corpus, embed=embed, vector_cache=cache)
        hits = index.search("widget", k=2)
        self.assertIn("dense", hits[0].lists)
        first = len(calls)
        kb.Index(self.corpus, embed=embed, vector_cache=cache)
        self.assertEqual(len(calls), first)  # nothing re-embedded

    def test_passages_stay_available_for_private_research(self):
        found = self.index.passages("widget demand 2030")
        self.assertEqual(found[0]["page"], 3)


class HorizonTests(BaseCase):
    def test_a_newer_report_revision_shows_beside_the_one_it_replaced(self):
        rows = self.index.timeline("widget demand")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 120)
        self.assertEqual(rows[0]["cite"], "Acme Research, Jun 2026, p. 3")
        self.assertEqual([h["value"] for h in rows[0]["history"]], [100])

    def test_the_timeline_respects_as_of(self):
        rows = self.index.timeline("widget demand", as_of=date(2025, 6, 1))
        self.assertEqual(rows[0]["value"], 100)
        self.assertEqual(rows[0]["history"], [])


class EvidenceTests(BaseCase):
    def test_claims_become_one_cited_article_per_source(self):
        articles = self.index.evidence_articles("Will widget demand pass 110 million units by 2030?")
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]["title"], "Annual Outlook (Acme Research, Jun 2026)")
        self.assertIn("120 million units", articles[0]["summary"])
        self.assertIn("Earlier: 100 million units (Acme Research, Jan 2025, p. 3).", articles[0]["summary"])
        self.assertNotIn("reach 100 million", articles[1]["summary"])  # the superseded projection is history only
        self.assertIn("(p. 3)", articles[0]["summary"])
        self.assertEqual(articles[0]["source_channel"], "knowledge_base")
        self.assertEqual(articles[0]["publish_date"], "2026-06-01")
        self.assertLessEqual(articles[0]["relevance_score"], 1.0)

    def test_a_question_the_base_knows_nothing_about_gets_nothing(self):
        self.assertEqual(self.index.evidence_articles("Who wins the Champions League final?"), [])

    def test_the_server_reads_the_base_only_when_configured(self):
        from analyzing_llm_rationale import server
        kb._shared.clear()
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root)}):
            found = server._knowledge_articles("Widget demand by 2030?")
            self.assertTrue(found)
            with mock.patch.dict(os.environ, {"KNOWLEDGE_EVIDENCE": "0"}):
                self.assertEqual(server._knowledge_articles("Widget demand by 2030?"), [])
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root / "missing")}):
            self.assertEqual(server._knowledge_articles("Widget demand by 2030?"), [])

    def test_a_base_with_errors_is_not_served(self):
        _write(self.root / "sources" / NEW / "relations.jsonl", [{"subject": "ghost", "predicate": "x",
                                                                  "object": "widgets"}])
        kb._shared.clear()
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root)}):
            self.assertIsNone(kb.shared_index())


class SharedIndexTests(BaseCase):
    """On the live server KNOWLEDGE_DIR is a Cloud Storage mount, so files are re-checked on a timer."""

    def setUp(self):
        super().setUp()
        kb._shared.clear()

    def tearDown(self):
        kb._shared.clear()
        super().tearDown()

    def _add_source(self, sid="acme-2026-09-gadgets"):
        folder = self.root / "sources" / sid
        _write(folder / "source.json", {"id": sid, "title": "Gadget Note", "publisher": "Acme Research",
                                        "published": "2026-09-01"})
        _write(folder / "claims.jsonl", [{"id": f"{sid}:c001", "kind": "fact", "page": 1,
                                          "text": "Gadget output doubled in 2026.", "entities": ["gadgets"]}])
        _write(folder / "relations.jsonl", [])
        _write(folder / "chunks.jsonl", [])
        return folder

    def test_files_are_not_rechecked_inside_the_window(self):
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root), "KNOWLEDGE_RESTAT_S": "300"}):
            first = kb.shared_index()
            self._add_source()
            with mock.patch.object(Path, "rglob", side_effect=AssertionError("re-stat inside the window")):
                self.assertIs(kb.shared_index(), first)
            self.assertEqual(len(first.corpus.sources), 2)

    def test_a_new_source_is_picked_up_once_the_window_passes(self):
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root), "KNOWLEDGE_RESTAT_S": "0"}):
            first = kb.shared_index()
            self.assertIs(kb.shared_index(), first)  # nothing changed: no rebuild
            self._add_source()
            second = kb.shared_index()
            self.assertIsNot(second, first)
            self.assertEqual(len(second.corpus.sources), 3)

    def test_a_removed_file_is_noticed_even_when_the_newest_mtime_is_unchanged(self):
        folder = self._add_source()
        for path in self.root.rglob("*.json*"):
            os.utime(path, (1_700_000_000, 1_700_000_000))
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root), "KNOWLEDGE_RESTAT_S": "0"}):
            first = kb.shared_index()
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
            second = kb.shared_index()
            self.assertIsNot(second, first)
            self.assertEqual(len(second.corpus.sources), 2)

    def test_a_bad_window_setting_falls_back_to_five_minutes(self):
        with mock.patch.dict(os.environ, {"KNOWLEDGE_RESTAT_S": "soon"}):
            self.assertEqual(kb._restat_seconds(), 300.0)

    def test_forecast_evidence_never_carries_report_passages(self):
        # /predict evidence is public; report passages are often licensed and stay in the base.
        passages = [c["text"] for c in self.corpus.chunks]
        self.assertTrue(passages)
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root)}):
            from analyzing_llm_rationale import server
            articles = server._knowledge_articles("Will widget demand pass 110 million units by 2030?")
        self.assertTrue(articles)
        for article in articles:
            for value in article.values():
                for passage in passages:
                    self.assertNotIn(passage, str(value))


def _sync_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "knowledge_r2_sync", Path(__file__).resolve().parents[1] / "scripts" / "knowledge_r2_sync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeR2:
    """head_object/get_object over one archive, like the private R2 bucket."""

    def __init__(self, data: bytes, etag: str = '"e1"'):
        self.data, self.etag, self.gets, self.fail = data, etag, 0, None

    def head_object(self, Bucket, Key):
        if self.fail:
            raise self.fail
        return {"ETag": self.etag, "ContentLength": len(self.data)}

    def get_object(self, Bucket, Key, IfMatch=None):
        self.gets += 1
        return {"Body": io.BytesIO(self.data)}


class RemoteBaseTests(BaseCase):
    """The live server reads the base from a private R2 archive (KNOWLEDGE_R2_*)."""

    def setUp(self):
        super().setUp()
        kb._shared.clear()
        self.pack = _sync_script().pack
        self.cache = Path(self.tmp.name) / "cache"
        self.r2 = FakeR2(self.pack(self.root))
        self.env = mock.patch.dict(os.environ, {
            "KNOWLEDGE_R2_ENDPOINT_URL": "https://example.r2.cloudflarestorage.com",
            "KNOWLEDGE_R2_BUCKET": "state", "KNOWLEDGE_R2_ACCESS_KEY_ID": "id",
            "KNOWLEDGE_R2_SECRET_ACCESS_KEY": "secret", "KNOWLEDGE_CACHE_DIR": str(self.cache),
            "KNOWLEDGE_RESTAT_S": "0"})
        self.env.start()
        os.environ.pop("KNOWLEDGE_DIR", None)
        self.client = mock.patch.object(kb, "_remote_client", return_value=self.r2)
        self.client.start()

    def tearDown(self):
        self.client.stop()
        self.env.stop()
        kb._shared.clear()
        super().tearDown()

    def _with_new_source(self):
        folder = self.root / "sources" / "acme-2026-09-gadgets"
        _write(folder / "source.json", {"id": "acme-2026-09-gadgets", "title": "Gadget Note",
                                        "publisher": "Acme Research", "published": "2026-09-01"})
        _write(folder / "claims.jsonl", [{"id": "acme-2026-09-gadgets:c001", "kind": "fact", "page": 1,
                                          "text": "Gadget output doubled in 2026.", "entities": ["gadgets"]}])
        return self.pack(self.root)

    def test_the_server_reads_the_r2_copy(self):
        index = kb.shared_index()
        self.assertEqual(len(index.corpus.sources), 2)
        from analyzing_llm_rationale import server
        self.assertTrue(server._knowledge_articles("Will widget demand pass 110 million units by 2030?"))

    def test_it_downloads_again_only_when_the_etag_changes(self):
        first = kb.shared_index()
        self.assertIs(kb.shared_index(), first)
        self.assertEqual(self.r2.gets, 1)
        self.r2.data, self.r2.etag = self._with_new_source(), '"e2"'
        second = kb.shared_index()
        self.assertEqual(self.r2.gets, 2)
        self.assertEqual(len(second.corpus.sources), 3)
        self.assertEqual(len(list(self.cache.iterdir())), 1)  # the old copy is gone

    def test_inside_the_window_r2_is_not_asked_at_all(self):
        os.environ["KNOWLEDGE_RESTAT_S"] = "300"
        first = kb.shared_index()
        self.r2.fail = AssertionError("asked R2 inside the window")
        self.assertIs(kb.shared_index(), first)

    def test_a_failed_fetch_keeps_the_last_good_index(self):
        first = kb.shared_index()
        self.r2.fail = ConnectionError("R2 is down")
        with self.assertLogs("analyzing_llm_rationale.knowledge", "WARNING"):
            self.assertIs(kb.shared_index(), first)

    def test_a_new_copy_with_errors_keeps_the_last_good_index(self):
        first = kb.shared_index()
        _write(self.root / "sources" / NEW / "relations.jsonl", [{"subject": "ghost", "predicate": "x",
                                                                  "object": "widgets"}])
        self.r2.data, self.r2.etag = self.pack(self.root), '"e2"'
        with self.assertLogs("analyzing_llm_rationale.knowledge", "WARNING"):
            self.assertIs(kb.shared_index(), first)

    def test_no_base_until_all_four_settings_are_there(self):
        del os.environ["KNOWLEDGE_R2_SECRET_ACCESS_KEY"]
        self.assertIsNone(kb.remote_config())
        self.assertIsNone(kb.shared_index())
        self.assertEqual(self.r2.gets, 0)

    def test_an_existing_local_folder_wins_over_r2(self):
        os.environ["KNOWLEDGE_DIR"] = str(self.root)
        self.assertIsNotNone(kb.shared_index())
        self.assertEqual(self.r2.gets, 0)

    def test_unsafe_archives_are_refused(self):
        import tarfile
        for name, kind in (("../escape.json", tarfile.REGTYPE), ("/abs.json", tarfile.REGTYPE),
                           ("link.json", tarfile.SYMTYPE)):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.SYMTYPE:
                    info.linkname = "/etc/passwd"
                    tar.addfile(info)
                else:
                    info.size = 2
                    tar.addfile(info, io.BytesIO(b"{}"))
            with self.subTest(name=name), tempfile.TemporaryDirectory() as out:
                with self.assertRaises(ValueError):
                    kb.unpack(buf.getvalue(), Path(out))

class AgentFeedTests(BaseCase):
    def test_a_brief_is_compact_cited_and_current(self):
        rows = self.index.brief("widget demand by 2030", k=2)
        self.assertEqual(rows[0]["source"], "Acme Research, Jun 2026, p. 3")
        self.assertEqual(rows[0]["horizon"], "2030")
        self.assertNotIn("100 million", " ".join(r["claim"] for r in rows))  # superseded projection left out
        self.assertEqual(self.index.brief("Who wins the Champions League final?"), [])

    def test_the_module_brief_is_empty_without_a_base(self):
        kb._shared.clear()
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": ""}):
            self.assertEqual(kb.brief("widget demand by 2030"), [])
        with mock.patch.dict(os.environ, {"KNOWLEDGE_DIR": str(self.root)}):
            self.assertTrue(kb.brief("widget demand by 2030"))

    def test_figures_filter_by_metric_place_and_date(self):
        self.assertEqual(len(self.index.figures("Widget Demand")), 2)
        self.assertEqual(len(self.index.figures("widget demand", geo="mars")), 0)
        early = self.index.figures("widget demand", as_of=date(2025, 6, 1))
        self.assertEqual([f["claim"]["value"] for f in early], [100])


class IngestTests(unittest.TestCase):
    def test_passages_carry_their_page_and_section_and_never_cross_a_page(self):
        pages = ["2.1. Demand grows fast\n" + "Widgets sell. " * 200, "", "Section 3. Policy\nTaxes rise."]
        chunks = kb.chunk_pages("s", pages, size=500, overlap=50)
        self.assertEqual({c["page"] for c in chunks}, {1, 3})
        self.assertTrue(all(len(c["text"]) <= 500 for c in chunks))
        self.assertEqual(chunks[0]["section"], "2.1. Demand grows fast")
        self.assertEqual(chunks[-1]["section"], "Section 3. Policy")
        self.assertEqual(len({c["id"] for c in chunks}), len(chunks))

    def test_a_claim_is_kept_only_if_its_numbers_are_in_the_passage(self):
        passage = "Demand reaches 2,354 TWh by 2035, about 7% of the total."
        self.assertTrue(kb.grounded({"text": "Demand hits 2,354 TWh in 2035.", "value": 2354}, passage))
        self.assertFalse(kb.grounded({"text": "Demand hits 2,500 TWh in 2035."}, passage))
        self.assertTrue(kb.grounded({"text": "About 7% by 2035.", "value": 7.0}, passage))

    def test_extraction_drops_invented_numbers_and_links_relations(self):
        chunk = {"id": "s:p4:1", "page": 4, "section": "2. Demand",
                 "text": "Widget demand reaches 120 million units in 2030, led by Ruritania."}
        reply = """<think>reasoning</think> Sure: {"claims": [
          {"kind": "projection", "text": "Widget demand reaches 120 million units by 2030.", "entities": ["Widgets"],
           "metric": "widget demand", "value": "120", "unit": "million units", "geo": "global", "horizon": 2030},
          {"kind": "projection", "text": "Widget demand reaches 150 million units by 2032.", "entities": ["Widgets"]},
          {"kind": "rumour", "text": "Something."}],
          "relations": [{"subject": "Ruritania", "predicate": "leads demand for", "object": "Widgets", "claim": 0}]}"""
        claims, relations = kb.extract(chunk, {"title": "T", "publisher": "P", "published": "2026-01-01"},
                                       lambda prompt: reply)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["value"], 120)
        self.assertEqual(claims[0]["horizon"], "2030")
        self.assertEqual(claims[0]["page"], 4)
        self.assertEqual(relations, [{"subject": "Ruritania", "predicate": "leads_demand_for", "object": "Widgets",
                                      "claim_index": 0}])
        self.assertEqual(kb.extract(chunk, {}, lambda prompt: "no json here"), ([], []))

    def test_entity_names_map_to_existing_ids_or_new_ones(self):
        entities = {"widgets": {"id": "widgets", "name": "Widgets", "aliases": ["widget"]}}
        ids = kb.merge_entities(entities, ["widget", "Gadget Co."])
        self.assertEqual(ids["widget"], "widgets")
        self.assertEqual(ids["Gadget Co."], "gadget_co")
        self.assertEqual(entities["gadget_co"]["type"], "concept")


class CliTests(unittest.TestCase):
    def test_add_validate_and_search_round_trip(self):
        import knowledge as cli
        with tempfile.TemporaryDirectory() as tmp:
            root = make_base(Path(tmp) / "base")
            text = Path(tmp) / "report.txt"
            text.write_text("1. Intro\nWidgets matter.\f2.1. Demand\nWidget demand reaches 130 million units.",
                            encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(["--dir", str(root), "add", str(text), "--id", "acme-2026-11-notes",
                                 "--title", "Widget Notes", "--publisher", "Acme Research",
                                 "--published", "2026-11-01"])
            self.assertEqual(code, 0, out.getvalue())
            chunks = (root / "sources" / "acme-2026-11-notes" / "chunks.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(c)["page"] for c in chunks], [1, 2])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["--dir", str(root), "search", "widget demand 2030", "--k", "1"])
            self.assertIn("Acme Research, Jun 2026, p. 3", out.getvalue())


if __name__ == "__main__":
    unittest.main()
