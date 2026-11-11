from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import evidence_time as et  # noqa: E402
from analyzing_llm_rationale.news_pipeline import NewsPipeline  # noqa: E402
from analyzing_llm_rationale.pipeline import build_user_prompt, extract_summary_items  # noqa: E402

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


def _rss(days_ago: float) -> str:
    """The date the way Google News's RSS writes it."""
    return (NOW - timedelta(days=days_ago)).strftime("%a, %d %b %Y %H:%M:%S GMT")


class DateTests(unittest.TestCase):
    def test_every_form_the_sources_use_parses(self):
        want = datetime(2026, 11, 10, 14, 0, tzinfo=timezone.utc)
        for text in ("Tue, 10 Nov 2026 14:00:00 GMT", "2026-11-10T14:00:00Z", "2026-11-10T15:00:00+01:00",
                     "20261110T140000Z", "20261110140000"):
            with self.subTest(text=text):
                self.assertEqual(et.parse_publish_date(text), want)
        self.assertEqual(et.parse_publish_date("2026-11-10"), datetime(2026, 11, 10, tzinfo=timezone.utc))
        for junk in ("", "  ", None, "yesterday", "20261399T000000Z", 5):
            with self.subTest(junk=junk):
                self.assertIsNone(et.parse_publish_date(junk))

    def test_undated_evidence_is_dated_by_its_link_or_title(self):
        cases = [
            ({"url": "https://news.usni.org/2026/06/18/u-s-ends-naval-blockade-of-iran"}, (2026, 6, 18), "link"),
            ({"url": "https://example.com/briefs/odds-fall-20261004-0001"}, (2026, 10, 4), "link"),
            ({"url": "https://understandingwar.org/research/middle-east/iran-update-november-9-2026/"}, (2026, 11, 9), "link"),
            ({"url": "https://example.com/a", "title": "Iran Update: Nov. 9, 2026 | ISW"}, (2026, 11, 9), "title"),
            ({"url": "https://example.com/a", "title": "Markets on 3 September 2026"}, (2026, 9, 3), "title"),
        ]
        for article, ymd, where in cases:
            with self.subTest(article=article):
                self.assertEqual(et.inferred_date(article, NOW), (datetime(*ymd, tzinfo=timezone.utc), where))
        for article in ({"url": "https://en.wikipedia.org/wiki/U.S._naval_blockade_of_Iranian_ports"},
                        {"url": "https://example.com/2027/01/05/preview"},  # in the future
                        {"url": "https://example.com/item/202611"}):
            with self.subTest(article=article):
                self.assertEqual(et.inferred_date(article, NOW), (None, ""))
        stated = {"publish_date": "2026-11-01", "url": "https://example.com/2026/06/18/x"}
        self.assertEqual(et.article_date(stated, NOW), (datetime(2026, 11, 1, tzinfo=timezone.utc), "published"))

    def test_the_last_30_days_count_in_full_and_older_evidence_fades_to_a_floor(self):
        self.assertEqual(et.recency_weight(NOW - timedelta(days=2), NOW), 1.0)
        self.assertEqual(et.recency_weight(NOW - timedelta(days=30), NOW), 1.0)
        self.assertAlmostEqual(et.recency_weight(NOW - timedelta(days=60), NOW), 0.5)
        self.assertEqual(et.recency_weight(NOW - timedelta(days=400), NOW), et.FLOOR)
        self.assertEqual(et.recency_weight(None, NOW), et.UNDATED_WEIGHT)
        self.assertEqual(et.recency_weight(NOW + timedelta(hours=3), NOW), 1.0)  # a clock a little ahead

    def test_bands(self):
        self.assertEqual(et.timeline_band(NOW - timedelta(days=3), NOW), et.BAND_LATEST)
        self.assertEqual(et.timeline_band(NOW - timedelta(days=12), NOW), et.BAND_RECENT)
        self.assertEqual(et.timeline_band(NOW - timedelta(days=45), NOW), et.BAND_OLDER)
        self.assertEqual(et.timeline_band(None, NOW), et.BAND_UNDATED)


class RankTests(unittest.TestCase):
    Q = "Will Sotheby's sell a van Gogh for over $150 million at the November auction?"

    def _pipeline(self):
        return NewsPipeline(use_query_planner=False, summarize_articles=False, use_embeddings=False)

    def test_recent_news_outranks_the_same_story_from_months_ago(self):
        old = {"title": "Sotheby's van Gogh estimate $150 million November auction", "url": "a", "publish_date": _rss(120)}
        new = {"title": "Sotheby's van Gogh estimate $150 million November auction", "url": "b", "publish_date": _rss(4)}
        ranked = self._pipeline().rank(self.Q, [old, new], now=NOW)
        self.assertEqual([a["url"] for a in ranked], ["b", "a"])
        self.assertEqual((ranked[0]["recency_weight"], ranked[0]["age_days"]), (1.0, 4.0))
        self.assertEqual(ranked[1]["recency_weight"], et.FLOOR)

    def test_an_undated_result_with_an_old_link_ranks_as_old(self):
        stale = {"title": "Sotheby's van Gogh $150 million November auction", "url": "https://x.com/2026/03/02/a"}
        fresh = {"title": "Sotheby's van Gogh $150 million November auction", "url": "https://y.com/b", "publish_date": _rss(6)}
        ranked = self._pipeline().rank(self.Q, [stale, fresh], now=NOW)
        self.assertEqual([a["url"] for a in ranked], ["https://y.com/b", "https://x.com/2026/03/02/a"])
        self.assertEqual((ranked[1]["dated_by"], ranked[1]["recency_weight"]), ("link", et.FLOOR))

    def test_within_the_window_relevance_still_decides(self):
        close = {"title": "Sotheby's van Gogh $150 million November auction estimate", "url": "a", "publish_date": _rss(25)}
        loose = {"title": "Auction season opens in New York", "url": "b", "publish_date": _rss(1)}
        ranked = self._pipeline().rank(self.Q, [loose, close], now=NOW)
        self.assertEqual(ranked[0]["url"], "a")


class PromptTimelineTests(unittest.TestCase):
    def _record(self, articles):
        return {"question": "Will an artwork sell for $150 million by December 31?", "current_time": NOW.isoformat(),
                "news_articles": articles}

    def test_rss_dates_sort_by_date_not_by_weekday(self):
        # "Wed, 21 Oct" sorts after "Tue, 10 Nov" as text; by date it is the oldest.
        items = extract_summary_items(self._record([
            {"title": "a", "publish_date": "Tue, 10 Nov 2026 14:00:00 GMT"},
            {"title": "b", "publish_date": "Wed, 21 Oct 2026 14:00:00 GMT"},
            {"title": "c", "publish_date": "2026-11-08T10:00:00Z"},
            {"title": "d"},
        ]), "full")
        self.assertEqual([i["title"] for i in items], ["a", "c", "b", "d"])

    def test_the_prompt_reads_as_a_dated_timeline(self):
        prompt = build_user_prompt(self._record([
            {"title": "Preview of the November sales", "source": "NYT", "publish_date": _rss(2)},
            {"title": "Autumn season outlook", "source": "Artnet", "publish_date": _rss(18)},
            {"title": "Spring record sale", "source": "FT", "publish_date": _rss(150)},
            {"title": "Auction house profile", "source": "Web"},
            {"title": "U.S. Ends Naval Blockade of Iran", "source": "USNI", "url": "https://news.usni.org/2026/06/18/x"},
        ]), "[question]\nReturn JSON.", "summary")
        order = [prompt.index(text) for text in (
            "Evidence timeline (newest first; the last 30 days carry the most signal):",
            f"-- {et.BAND_LATEST} --", "Preview of the November sales",
            f"-- {et.BAND_RECENT} --", "Autumn season outlook",
            f"-- {et.BAND_OLDER} --", "Spring record sale",
            f"-- {et.BAND_UNDATED} --", "Auction house profile")]
        self.assertEqual(order, sorted(order))
        self.assertIn(f"NYT: Preview of the November sales [{(NOW - timedelta(days=2)).date().isoformat()}]", prompt)
        self.assertIn("[⚠ 150d old]", prompt)
        self.assertIn("USNI: U.S. Ends Naval Blockade of Iran [2026-06-18, dated by its link] [⚠ 146d old]", prompt)
        self.assertIn("Evidence item(s) #3, #4 are more than 30 days old", prompt)
        self.assertLess(prompt.index("U.S. Ends Naval Blockade"), prompt.index(f"-- {et.BAND_UNDATED} --"))
        self.assertNotIn("#2 are more", prompt)  # 18 days is inside the window


if __name__ == "__main__":
    unittest.main()
