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

        def fake_parse(url):
            calls.append(url)
            return SimpleNamespace(
                entries=[
                    {
                        "title": "Google News result",
                        "link": "https://example.com/google-news",
                        "published": "Fri, 29 May 2026 12:00:00 GMT",
                        "summary": "A relevant article summary.",
                    }
                ]
            )

        original = sys.modules.get("feedparser")
        sys.modules["feedparser"] = SimpleNamespace(parse=fake_parse)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._fetch_google_news("Federal Reserve rate cut", limit=5)
        finally:
            if original is None:
                sys.modules.pop("feedparser", None)
            else:
                sys.modules["feedparser"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "Google News")
        self.assertEqual(articles[0]["url"], "https://example.com/google-news")
        self.assertIn("news.google.com/rss/search", calls[0])
        self.assertIn("Federal+Reserve+rate+cut", calls[0])

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
