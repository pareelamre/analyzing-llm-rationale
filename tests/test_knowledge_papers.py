from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import knowledge_papers as papers  # noqa: E402
from analyzing_llm_rationale import knowledge_reports as reports  # noqa: E402
from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402

TODAY = date(2026, 11, 30)

RSS = """<?xml version='1.0' encoding='UTF-8'?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
<channel><title>cs.LG updates on arXiv.org</title>
<item><title>Calibrated forecasts from language model agents on prediction markets</title>
  <link>https://arxiv.org/abs/2611.01234</link>
  <description>arXiv:2611.01234v1 Announce Type: new
Abstract: We study LLM agents that forecast Kalshi and Polymarket questions and measure their Brier score.</description>
  <guid isPermaLink="false">oai:arXiv.org:2611.01234v1</guid>
  <category>cs.LG</category><arxiv:announce_type>new</arxiv:announce_type>
  <dc:rights>http://creativecommons.org/licenses/by/4.0/</dc:rights>
  <dc:creator>Ada Lovelace, Alan Turing</dc:creator></item>
<item><title>A faster sorting network</title>
  <link>https://arxiv.org/abs/2611.05678</link>
  <description>arXiv:2611.05678v1 Announce Type: new
Abstract: We sort things quickly.</description>
  <category>cs.LG</category><arxiv:announce_type>new</arxiv:announce_type>
  <dc:creator>Someone Else</dc:creator></item>
<item><title>Forecasting with an old trick</title>
  <link>https://arxiv.org/abs/2610.09999</link>
  <description>arXiv:2610.09999v2 Announce Type: replace
Abstract: A new version of a forecasting paper.</description>
  <arxiv:announce_type>replace</arxiv:announce_type></item>
</channel></rss>"""


def _work(**kw):
    base = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1016/j.ijforecast.2026.10.003",
            "display_name": "Coherent forecast combination for prediction markets", "publication_date": "2026-11-20",
            "type": "article", "authorships": [{"author": {"display_name": "Daniele Girolimetto"}}],
            "primary_location": {"source": {"display_name": "International Journal of Forecasting"}},
            "best_oa_location": {"pdf_url": "https://example.org/paper.pdf", "license": "cc-by"},
            "abstract_inverted_index": {"We": [0], "combine": [1], "forecasts": [2]}}
    base.update(kw)
    return base


class CardTests(unittest.TestCase):
    def test_arxiv_feed_keeps_forecasting_papers_and_skips_replacements(self):
        cards = papers.parse_arxiv_rss(RSS, "cs.LG", "forecast", TODAY.isoformat())
        self.assertEqual([c["id"] for c in cards], ["arxiv:2611.01234"])
        card = cards[0]
        self.assertEqual(card["pdf_url"], "https://arxiv.org/pdf/2611.01234")
        self.assertEqual(card["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertIn("prediction-markets", card["topics"])
        self.assertTrue(card["abstract"].startswith("We study LLM agents"))
        everything = papers.parse_arxiv_rss(RSS, "q-fin", "all", TODAY.isoformat())
        self.assertEqual(len(everything), 2)  # quantitative finance keeps every new paper

    def test_openalex_cards_rebuild_the_abstract_and_map_arxiv_dois(self):
        card = papers.openalex_card(_work(), TODAY.isoformat())
        self.assertEqual(card["id"], "doi:10.1016/j.ijforecast.2026.10.003")
        self.assertEqual(card["abstract"], "We combine forecasts")
        arxiv = papers.openalex_card(_work(doi="https://doi.org/10.48550/arXiv.2611.01234"), TODAY.isoformat())
        self.assertEqual(arxiv["id"], "arxiv:2611.01234")

    def test_unreviewed_foreign_and_non_paper_works_are_left_out(self):
        zenodo = _work(primary_location={"source": {"display_name": "Zenodo (CERN European Organization for Nuclear Research)"}})
        self.assertIsNone(papers.openalex_card(zenodo, TODAY.isoformat()))
        self.assertIsNone(papers.openalex_card(_work(display_name="概率判断与群体智慧"), TODAY.isoformat()))
        self.assertIsNone(papers.openalex_card(_work(type="dataset"), TODAY.isoformat()))
        bare = _work(abstract_inverted_index=None, primary_location={"source": {"display_name": "Some Blog"}})
        self.assertIsNone(papers.openalex_card(bare, TODAY.isoformat()))

    def test_score_is_led_by_the_main_topic_not_the_keyword_count(self):
        focused = papers.make_card(id="a", title="Prediction markets and the Brier score", abstract="",
                                   pdf_url="x", source="arxiv")
        soup = papers.make_card(id="b", title="Sports, elections, weather, epidemics, inflation and India", abstract="")
        self.assertGreater(focused["score"], soup["score"])
        self.assertEqual(focused["score"], 5 + 2 + 1 + 1)
        # Fitting a pricing model is not forecast calibration.
        self.assertNotIn("calibration-scoring", papers.topics("We calibrate the Heston model to option prices."))


class WeeklyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cards = [
            papers.make_card(id="arxiv:1", title="LLM forecasting on prediction markets", abstract="", source="arxiv",
                             authors=["A"], pdf_url="https://arxiv.org/pdf/1", url="https://arxiv.org/abs/1",
                             venue="arXiv", found="2026-11-29"),
            papers.make_card(id="arxiv:2", title="More prediction markets and calibration error", abstract="",
                             source="arxiv", authors=["A"], pdf_url="https://arxiv.org/pdf/2", venue="arXiv",
                             found="2026-11-29"),  # same lead author: one paper per author a week
            papers.make_card(id="doi:3", title="Pairs trading with cointegration in commodities", abstract="",
                             authors=["B"], pdf_url="https://x/3.pdf", venue="Quantitative Finance",
                             published="2026-11-20", found="2026-11-29"),
            papers.make_card(id="doi:4", title="Dengue forecasting with neural networks", abstract="",
                             authors=["C"], pdf_url="https://x/4.pdf", venue="Epidemics", found="2026-11-29"),
            papers.make_card(id="doi:5", title="Prediction markets, no PDF", abstract="", authors=["D"],
                             venue="Electoral Studies", found="2026-11-29"),
        ]
        folder = self.root / "papers" / "cards"
        folder.mkdir(parents=True)
        (folder / "2026-11.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cards))

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_best_core_papers_with_a_pdf_are_queued_once(self):
        out = papers.weekly(self.root, TODAY)
        self.assertEqual(out["queued"], ["arxiv:1", "doi:3"])
        rows = [json.loads(line) for line in (self.root / "inbox.jsonl").read_text().splitlines()]
        self.assertEqual({r["kind"] for r in rows}, {"report"})
        self.assertEqual(rows[1]["files"][0]["content_type"], "application/pdf")
        self.assertEqual(reports._published(rows[1]), ("2026-11-20", False))
        digest = (self.root / "papers" / "digest" / "latest.md").read_text()
        self.assertIn("could inform", digest)
        self.assertTrue((self.root / "papers" / "digest" / "2026-W49.md").exists())
        self.assertEqual(papers.weekly(self.root, TODAY)["queued"], ["arxiv:2"])  # the rest wait their turn

    def test_the_reader_queues_a_paper_from_the_inbox(self):
        papers.weekly(self.root, TODAY)
        jobs = reports.from_inbox(self.root)
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(j["id"].startswith("papers-") for j in jobs))
        self.assertTrue(all(j.get("file_url") for j in jobs))


class CollectTests(unittest.TestCase):
    def test_collect_adds_each_paper_once_across_feeds_and_days(self):
        def get(url, *, max_bytes=0):
            if url.endswith("/robots.txt"):
                return watch.Response(404, b"", "text/plain", url, None)
            if "rss.arxiv.org" in url:
                return watch.Response(200, RSS.encode(), "application/rss+xml", url, None)
            work = _work(doi="https://doi.org/10.48550/arXiv.2611.01234")  # the arXiv paper again
            body = json.dumps({"results": [work, _work()], "meta": {"next_cursor": None}})
            return watch.Response(200, body.encode(), "application/json", url, None)
        crawler = watch.Crawler(get, pause=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = papers.collect(root, crawler, TODAY)
            ids = sorted(c["id"] for c in first["found"])
            self.assertEqual(ids, ["arxiv:2611.01234", "arxiv:2611.05678", "doi:10.1016/j.ijforecast.2026.10.003"])
            self.assertEqual(papers.collect(root, crawler, TODAY)["found"], [])
            self.assertEqual(len(papers.all_cards(root)), 3)


if __name__ == "__main__":
    unittest.main()
