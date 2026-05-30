from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.news_pipeline import (  # noqa: E402
    NewsPipeline,
    _keyword_search_query,
    _lexical_relevance,
)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "articles": [
                {
                    "title": "Fed officials discuss rate path",
                    "url": "https://example.com/fed",
                    "seendate": "20260529120000",
                    "domain": "example.com",
                }
            ]
        }


class NewsPipelineSourceTests(unittest.TestCase):
    def test_fetch_gdelt_maps_doc_api_articles(self):
        calls = []

        def fake_get(url, params, timeout):
            calls.append((url, params, timeout))
            return FakeResponse()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._fetch_gdelt("Federal Reserve rate cut", limit=3)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "example.com")
        self.assertEqual(articles[0]["summary"], "Fed officials discuss rate path")
        self.assertEqual(calls[0][1]["query"], "Federal Reserve rate cut")
        self.assertEqual(calls[0][1]["maxrecords"], 3)

    def test_fetch_google_news_uses_search_rss(self):
        calls = []

        class FakeGoogleResponse:
            content = b"""
            <rss>
              <channel>
                <item>
                  <title>Google News result</title>
                  <link>https://example.com/google-news</link>
                  <pubDate>Fri, 29 May 2026 12:00:00 GMT</pubDate>
                  <description>A relevant article summary.</description>
                  <source>Example Source</source>
                </item>
              </channel>
            </rss>
            """

            def raise_for_status(self):
                return None

        def fake_get(url, headers, timeout):
            calls.append((url, headers, timeout))
            return FakeGoogleResponse()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._fetch_google_news("Federal Reserve rate cut", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "Example Source")
        self.assertEqual(articles[0]["url"], "https://example.com/google-news")
        self.assertIn("news.google.com/rss/search", calls[0][0])
        self.assertIn("Federal+Reserve+rate+cut", calls[0][0])

    def test_fetch_stooq_maps_static_rss_feeds(self):
        calls = []

        class FakeStooqResponse:
            content = b"""
            <rss>
              <channel>
                <title>Stooq - Wiadomosci Biznes</title>
                <item>
                  <title>Stooq market update</title>
                  <link>https://stooq.com/n/?f=123</link>
                  <pubDate>Sat, 30 May 2026 12:00:00 GMT</pubDate>
                  <description>A financial market update.</description>
                </item>
              </channel>
            </rss>
            """

            def raise_for_status(self):
                return None

        def fake_get(url, headers, timeout):
            calls.append((url, headers, timeout))
            return FakeStooqResponse()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._fetch_stooq(limit=1)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "Stooq")
        self.assertEqual(articles[0]["title"], "Stooq market update")
        self.assertEqual(articles[0]["summary"], "A financial market update.")
        self.assertEqual(articles[0]["search_query"], "Stooq - Wiadomosci Biznes")
        self.assertIn("static.stooq.com/rss/pl/b.rss", calls[0][0])

    def test_fetch_queries_all_configured_sources_before_dedupe(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._newsapi_key = None
        pipeline._fetch_sources = ("gdelt", "google-news", "stooq")
        calls = []

        def fake_gdelt(query, limit):
            calls.append(("gdelt", query, limit))
            return [{"title": "GDELT result", "url": "https://example.com/gdelt"}]

        def fake_google(query, limit):
            calls.append(("google-news", query, limit))
            return [{"title": "Google result", "url": "https://example.com/google"}]

        def fake_stooq(limit):
            calls.append(("stooq", limit))
            return [{"title": "Stooq result", "url": "https://example.com/stooq"}]

        pipeline._fetch_gdelt = fake_gdelt
        pipeline._fetch_google_news = fake_google
        pipeline._fetch_stooq = fake_stooq

        articles = pipeline.fetch("Federal Reserve rate cut", top_k=1)

        self.assertEqual([call[0] for call in calls], ["gdelt", "google-news", "stooq"])
        self.assertEqual(len(articles), 3)

    def test_select_diverse_sources_keeps_gdelt_google_and_stooq_when_available(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources = ("gdelt", "google-news", "stooq")
        ranked = [
            {"title": "Stooq 1", "url": "https://example.com/s1", "source_channel": "stooq"},
            {"title": "Stooq 2", "url": "https://example.com/s2", "source_channel": "stooq"},
            {"title": "GDELT", "url": "https://example.com/g", "source_channel": "gdelt"},
            {"title": "Google", "url": "https://example.com/n", "source_channel": "google-news"},
        ]

        selected = pipeline.select_diverse_sources(ranked, top_k=3)

        self.assertEqual(
            {article["source_channel"] for article in selected},
            {"gdelt", "google-news", "stooq"},
        )

    def test_rank_can_use_lightweight_lexical_scores(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._use_embeddings = False
        pipeline._embeddings = None
        articles = [
            {"title": "Sports update", "summary": "A tennis result."},
            {"title": "Federal Reserve rate cut", "summary": "Fed officials discuss rates."},
        ]

        ranked = pipeline.rank("Federal Reserve rate cut", articles)

        self.assertEqual(ranked[0]["title"], "Federal Reserve rate cut")
        self.assertGreater(ranked[0]["relevance_score"], ranked[1]["relevance_score"])
        self.assertGreater(_lexical_relevance("Federal Reserve rate cut", ranked[0]["title"]), 0)

    def test_retrieval_only_pipeline_does_not_require_llm_client(self):
        pipeline = NewsPipeline(
            api_key=None,
            use_query_planner=False,
            summarize_articles=False,
            use_embeddings=False,
            fetch_sources=("rss",),
        )

        self.assertIsNone(pipeline._llm)
        self.assertEqual(pipeline.plan_search_query("Will X happen?"), "Will X happen?")

    def test_keyword_search_query_removes_forecast_filler(self):
        query = _keyword_search_query(
            "Will the Federal Reserve cut US interest rates before July 31, 2026?"
        )

        self.assertIn("Federal", query)
        self.assertIn("Reserve", query)
        self.assertIn("interest", query)
        self.assertNotIn("Will", query)
        self.assertNotIn("before", query)


if __name__ == "__main__":
    unittest.main()
