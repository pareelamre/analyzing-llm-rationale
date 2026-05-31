from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.news_pipeline import (  # noqa: E402
    NewsPipeline,
    _is_finance_query,
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

    def test_fetch_web_uses_tavily_when_keyed(self):
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"results": [
                    {"title": "US CPI rises", "url": "https://ex.com/cpi", "content": "Inflation update."},
                ]}

        calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            calls.append((url, json, headers))
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(post=fake_post)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            pipeline._tavily_key = "tvly-key"
            articles = pipeline._fetch_web("US CPI inflation December 2026", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "ex.com")
        self.assertEqual(articles[0]["source_channel"], "web")
        self.assertEqual(articles[0]["url"], "https://ex.com/cpi")
        self.assertIn("api.tavily.com", calls[0][0])
        self.assertEqual(calls[0][1]["api_key"], "tvly-key")

    def test_fetch_web_uses_serper_when_keyed(self):
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"organic": [
                    {"title": "CPI report", "link": "https://news.com/cpi", "snippet": "Prices up."},
                ]}

        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(post=fake_post)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            pipeline._tavily_key = None
            pipeline._serper_key = "serper-key"
            articles = pipeline._fetch_web("US CPI", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["url"], "https://news.com/cpi")
        self.assertEqual(articles[0]["source"], "news.com")

    def test_stooq_not_in_default_sources(self):
        from analyzing_llm_rationale.news_pipeline import DEFAULT_FETCH_SOURCES
        self.assertNotIn("stooq", DEFAULT_FETCH_SOURCES)
        self.assertIn("web", DEFAULT_FETCH_SOURCES)

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

    def test_stooq_only_fetched_for_finance_questions(self):
        self.assertTrue(_is_finance_query("Will the Fed cut interest rates in 2026?"))
        self.assertTrue(_is_finance_query("Where will the S&P 500 close?"))
        self.assertFalse(_is_finance_query("can i talk to you?"))
        self.assertFalse(_is_finance_query("Will it rain in Berlin tomorrow?"))

        def make_pipeline():
            p = NewsPipeline.__new__(NewsPipeline)
            p._newsapi_key = None
            p._fetch_sources = ("gdelt", "stooq")
            p._fetch_gdelt = lambda query, limit: []
            return p

        calls = []
        finance = make_pipeline()
        finance._fetch_stooq = lambda limit: calls.append("finance") or []
        finance.fetch("Will the Fed cut interest rates?", top_k=5)
        self.assertEqual(calls, ["finance"])

        calls.clear()
        casual = make_pipeline()
        casual._fetch_stooq = lambda limit: calls.append("casual") or []
        casual.fetch("can i talk to you?", top_k=5)
        self.assertEqual(calls, [])  # Stooq not fetched for a non-finance question

    def test_select_diverse_drops_low_relevance_generic_sources(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources = ("gdelt", "stooq", "rss")
        ranked = [
            {"title": "GDELT", "url": "https://e.com/g", "source_channel": "gdelt", "relevance": 0.40},
            {"title": "Stooq junk", "url": "https://e.com/s", "source_channel": "stooq", "relevance": 0.0},
            {"title": "RSS junk", "url": "https://e.com/r", "source_channel": "rss", "relevance": 0.02},
        ]

        selected = pipeline.select_diverse_sources(ranked, top_k=5)
        channels = {a["source_channel"] for a in selected}

        self.assertIn("gdelt", channels)
        self.assertNotIn("stooq", channels)
        self.assertNotIn("rss", channels)

    def test_select_diverse_keeps_relevant_generic_sources(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources = ("gdelt", "stooq")
        ranked = [
            {"title": "GDELT", "url": "https://e.com/g", "source_channel": "gdelt", "relevance": 0.40},
            {"title": "Relevant Stooq", "url": "https://e.com/s", "source_channel": "stooq", "relevance": 0.45},
        ]

        selected = pipeline.select_diverse_sources(ranked, top_k=5)

        self.assertEqual({a["source_channel"] for a in selected}, {"gdelt", "stooq"})

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
        self.assertEqual(pipeline.plan_search_queries("Will X happen?"), ["Will X happen?"])

    def test_fetch_summarize_rank_uses_decomposed_search_queries(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._summarize_articles = False
        calls = []

        def fake_fetch(query, top_k):
            calls.append((query, top_k))
            return [
                {
                    "title": f"{query} article",
                    "summary": f"{query} summary",
                    "url": f"https://example.com/{query.replace(' ', '-')}",
                }
            ]

        pipeline.plan_search_queries = lambda question: [
            "Federal Reserve rate cut September 2026",
            "Federal Reserve inflation base rates 2026",
        ]
        pipeline.fetch = fake_fetch
        pipeline.rank = lambda question, articles: articles
        pipeline.select_diverse_sources = lambda ranked, top_k: ranked[:top_k]

        articles = pipeline.fetch_summarize_rank(
            "Will the Federal Reserve cut rates before September 30, 2026?",
            top_k=2,
        )

        self.assertEqual(
            [call[0] for call in calls],
            [
                "Federal Reserve rate cut September 2026",
                "Federal Reserve inflation base rates 2026",
            ],
        )
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]["search_query"], "Federal Reserve rate cut September 2026")

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
