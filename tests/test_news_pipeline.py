from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.news_pipeline import (  # noqa: E402
    NewsPipeline,
    _bm25_scores,
    _is_finance_query,
    _keyword_search_query,
    _lexical_relevance,
    _rrf_fuse,
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

    def test_fetch_uses_keyless_web_fallback_without_provider_credentials(self):
        pipeline = NewsPipeline(
            api_key=None,
            use_query_planner=False,
            summarize_articles=False,
            use_embeddings=False,
            fetch_sources=("web",),
        )
        article = {
            "title": "Cabinet departure report",
            "url": "https://example.com/cabinet",
            "source_channel": "web",
        }
        pipeline._fetch_web = mock.Mock(return_value=[article])

        articles = pipeline.fetch("Trump Cabinet departure", top_k=5)

        pipeline._fetch_web.assert_called_once_with(
            "Trump Cabinet departure",
            limit=10,
        )
        self.assertEqual(articles, [article])

    def test_duckduckgo_fallback_uses_get(self):
        class FakeResp:
            text = (
                '<div class="result">'
                '<a class="result__a" href="https://example.com/cabinet">'
                "Cabinet departure report</a>"
                '<div class="result__snippet">A cabinet official resigned.</div>'
                "</div>"
            )

            def raise_for_status(self):
                return None

        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._web_duckduckgo("Trump Cabinet departure", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Cabinet departure report")
        self.assertEqual(calls[0][1]["params"], {"q": "Trump Cabinet departure"})
        self.assertEqual(calls[0][1]["headers"], {"User-Agent": "Foresea/1.0"})
        self.assertTrue(calls[0][1]["allow_redirects"])

    def test_duckduckgo_fallback_uses_lite_when_html_is_empty(self):
        class FakeResp:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        lite_html = (
            "<table>"
            "<tr><td>1.</td><td>"
            '<a class="result-link" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fcabinet">'
            "Cabinet departure report</a></td></tr>"
            '<tr><td></td><td class="result-snippet">'
            "A cabinet official resigned.</td></tr>"
            "</table>"
        )
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            return FakeResp("" if "html.duckduckgo.com" in url else lite_html)

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._web_duckduckgo("Trump Cabinet departure", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(
            calls,
            [
                "https://html.duckduckgo.com/html/",
                "https://lite.duckduckgo.com/lite/",
            ],
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["url"], "https://example.com/cabinet")
        self.assertEqual(articles[0]["summary"], "A cabinet official resigned.")

    def test_duckduckgo_fallback_uses_lite_when_html_request_fails(self):
        class FakeResp:
            text = (
                '<a class="result-link" href="https://example.com/cabinet">'
                "Cabinet departure report</a>"
            )

            def raise_for_status(self):
                return None

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if "html.duckduckgo.com" in url:
                raise RuntimeError("primary endpoint timed out")
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._web_duckduckgo("Trump Cabinet departure", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "Cabinet departure report")

    def test_fetch_web_falls_back_when_configured_provider_is_empty(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._searxng_url = None
        pipeline._tavily_key = "tvly-key"
        pipeline._serper_key = None
        pipeline._brave_key = None
        pipeline._web_tavily = mock.Mock(return_value=[])
        fallback = [{"title": "Fallback result"}]
        pipeline._web_duckduckgo = mock.Mock(return_value=fallback)

        articles = pipeline._fetch_web("Trump Cabinet departure", limit=5)

        self.assertEqual(articles, fallback)
        pipeline._web_tavily.assert_called_once_with("Trump Cabinet departure", 5)
        pipeline._web_duckduckgo.assert_called_once_with("Trump Cabinet departure", 5)

    def test_fetch_web_uses_ap_news_when_duckduckgo_is_empty(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._searxng_url = None
        pipeline._tavily_key = None
        pipeline._serper_key = None
        pipeline._brave_key = None
        pipeline._web_duckduckgo = mock.Mock(return_value=[])
        fallback = [{"title": "Associated Press result"}]
        pipeline._web_ap_news = mock.Mock(return_value=fallback)

        articles = pipeline._fetch_web("Trump Cabinet departure", limit=5)

        self.assertEqual(articles, fallback)
        pipeline._web_ap_news.assert_called_once_with("Trump Cabinet departure", 5)

    def test_fetch_web_filters_out_blocked_social_media_domains(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._searxng_url = None
        pipeline._tavily_key = "tvly-key"
        pipeline._serper_key = None
        pipeline._brave_key = None
        pipeline._web_tavily = mock.Mock(return_value=[
            {"title": "Reddit thread speculating on the outcome", "source": "reddit.com"},
            {"title": "www.reddit.com mirror", "source": "www.reddit.com"},
            {"title": "Reuters wire report", "source": "reuters.com"},
        ])

        articles = pipeline._fetch_web("Trump Cabinet departure", limit=5)

        self.assertEqual([a["source"] for a in articles], ["reuters.com"])

    def test_fetch_web_falls_through_when_provider_is_entirely_blocked_domains(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._searxng_url = None
        pipeline._tavily_key = "tvly-key"
        pipeline._serper_key = None
        pipeline._brave_key = None
        pipeline._web_tavily = mock.Mock(return_value=[
            {"title": "Reddit thread", "source": "reddit.com"},
        ])
        fallback = [{"title": "Fallback result", "source": "apnews.com"}]
        pipeline._web_duckduckgo = mock.Mock(return_value=fallback)

        articles = pipeline._fetch_web("Trump Cabinet departure", limit=5)

        self.assertEqual(articles, fallback)
        pipeline._web_duckduckgo.assert_called_once_with("Trump Cabinet departure", 5)

    def test_ap_news_fallback_parses_search_results(self):
        class FakeResp:
            text = (
                '<div class="PageList-items-item">'
                '<div class="PagePromo">'
                '<div class="PagePromo-title">'
                '<a href="https://apnews.com/article/cabinet-departure">'
                "Trump Cabinet departure reported</a></div>"
                '<div class="PagePromo-description">'
                "The secretary resigned from the Cabinet.</div>"
                "</div></div>"
                '<div class="PageList-items-item">'
                '<div class="PagePromo">'
                '<div class="PagePromo-title">'
                '<a href="https://apnews.com/article/august-vote">'
                "August referendum scheduled</a></div>"
                '<div class="PagePromo-description">'
                "Voters will decide on membership talks.</div>"
                "</div></div>"
            )

            def raise_for_status(self):
                return None

        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            articles = pipeline._web_ap_news("Trump Cabinet departure", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "Associated Press")
        self.assertEqual(
            articles[0]["summary"],
            "The secretary resigned from the Cabinet.",
        )
        self.assertNotIn("August referendum scheduled", str(articles))
        self.assertLessEqual(len(articles), 3)
        self.assertEqual(calls[0][1]["params"], {"q": "Trump Cabinet departure"})

    def test_fetch_web_prefers_searxng(self):
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"results": [
                    {"title": "CPI", "url": "https://sx.com/cpi", "content": "Inflation."},
                ]}

        calls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append(url)
            return FakeResp()

        original = sys.modules.get("requests")
        sys.modules["requests"] = SimpleNamespace(get=fake_get)
        try:
            pipeline = NewsPipeline.__new__(NewsPipeline)
            # SearXNG wins even when other providers are also configured.
            pipeline._searxng_url = "https://searx.example/"
            pipeline._tavily_key = "tvly-key"
            pipeline._serper_key = "serper-key"
            articles = pipeline._fetch_web("US CPI", limit=5)
        finally:
            if original is None:
                sys.modules.pop("requests", None)
            else:
                sys.modules["requests"] = original

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["url"], "https://sx.com/cpi")
        self.assertIn("searx.example/search", calls[0])

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

    def test_fetch_dedupes_same_headline_across_syndicated_urls(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._newsapi_key = None
        pipeline._fetch_sources = ("gdelt", "google-news")
        title = "Federal Reserve signals September rate decision"

        pipeline._fetch_gdelt = lambda query, limit: [{
            "title": title,
            "url": "https://example.com/fed-september-rate-decision",
            "source_channel": "gdelt",
        }]
        pipeline._fetch_google_news = lambda query, limit: [{
            "title": f"{title} - Example News",
            "url": "https://news.example.com/story/123",
            "source_channel": "google-news",
        }]

        articles = pipeline.fetch("Federal Reserve rate decision", top_k=5)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source_channel"], "gdelt")

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

    def test_select_diverse_tries_next_channel_article_after_floor_rejection(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources = ("gdelt", "rss")
        pipeline._min_relevance = 0.3
        ranked = [
            {"title": "GDELT", "url": "https://e.com/g", "source_channel": "gdelt", "relevance": 0.50},
            {"title": "RSS junk", "url": "https://e.com/r1", "source_channel": "rss", "relevance": 0.02},
            {"title": "RSS relevant", "url": "https://e.com/r2", "source_channel": "rss", "relevance": 0.45},
        ]

        selected = pipeline.select_diverse_sources(ranked, top_k=3)

        self.assertEqual([article["url"] for article in selected], ["https://e.com/g", "https://e.com/r2"])

    def test_select_diverse_keeps_relevant_generic_sources(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._fetch_sources = ("gdelt", "stooq")
        ranked = [
            {"title": "GDELT", "url": "https://e.com/g", "source_channel": "gdelt", "relevance": 0.40},
            {"title": "Relevant Stooq", "url": "https://e.com/s", "source_channel": "stooq", "relevance": 0.45},
        ]

        selected = pipeline.select_diverse_sources(ranked, top_k=5)

        self.assertEqual({a["source_channel"] for a in selected}, {"gdelt", "stooq"})

    def test_rank_relevance_floor_fails_open_to_lexical_match_when_dense_is_weak(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._use_embeddings = True
        pipeline._embeddings = None
        pipeline._embed_fn = lambda texts: [
            [1.0, 0.0],
            [0.1, 0.99],
            [0.0, 1.0],
        ]
        pipeline._rerank_fn = None
        pipeline._fetch_sources = ("google-news",)
        pipeline._min_relevance = 0.25
        articles = [
            {"title": "Federal Reserve rate cut expected in 2026", "summary": "Fed may cut rates.", "source_channel": "google-news"},
            {"title": "Unrelated sports headline", "summary": "A tennis result.", "source_channel": "google-news"},
        ]

        ranked = pipeline.rank("Will the Federal Reserve cut interest rates before December 31, 2026?", articles)
        selected = pipeline.select_diverse_sources(ranked, top_k=5)

        self.assertGreaterEqual(ranked[0]["lexical_relevance"], 0.25)
        self.assertLess(ranked[0]["semantic_relevance"], 0.25)
        self.assertGreaterEqual(ranked[0]["relevance"], 0.25)
        self.assertEqual(len(selected), 1)
        self.assertIn("Federal Reserve", selected[0]["title"])

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

    def test_fetch_summarize_rank_retries_when_initial_candidates_are_filtered(self):
        pipeline = NewsPipeline.__new__(NewsPipeline)
        pipeline._summarize_articles = False
        question = "Will any member of Trump's Cabinet leave before August 2026?"
        initial_query = "Trump Cabinet August 2026"
        fallback_query = _keyword_search_query(question)
        calls = []

        def fake_fetch(query, top_k):
            calls.append((query, top_k))
            if query == initial_query:
                return [{"title": "Unrelated sports result", "url": "https://example.com/sport"}]
            return [
                {
                    "title": "Trump Cabinet departure",
                    "url": "https://example.com/cabinet",
                }
            ]

        def fake_rank(_question, articles):
            return [
                {
                    **article,
                    "relevance": 0.9 if "Cabinet departure" in article["title"] else 0.0,
                }
                for article in articles
            ]

        pipeline.plan_search_queries = lambda _question: [initial_query]
        pipeline.fetch = fake_fetch
        pipeline.rank = fake_rank
        pipeline.select_diverse_sources = lambda ranked, top_k: [
            article for article in ranked if article["relevance"] >= 0.25
        ][:top_k]

        articles = pipeline.fetch_summarize_rank(question, top_k=3)

        self.assertEqual(
            calls,
            [(initial_query, 5), (fallback_query, 10)],
        )
        self.assertEqual([article["title"] for article in articles], ["Trump Cabinet departure"])
        self.assertEqual(articles[0]["search_query"], fallback_query)

    def test_keyword_search_query_removes_forecast_filler(self):
        query = _keyword_search_query(
            "Will the Federal Reserve cut US interest rates before July 31, 2026?"
        )

        self.assertIn("Federal", query)
        self.assertIn("Reserve", query)
        self.assertIn("interest", query)
        self.assertNotIn("Will", query)
        self.assertNotIn("before", query)


class RelevanceFilterTests(unittest.TestCase):
    Q = "Will the San Antonio Spurs win the 2026 NBA Finals?"

    def test_lexical_relevance_ignores_stopwords(self):
        # Irrelevant junk (no topical overlap) scores 0 — not nonzero for "the".
        self.assertEqual(_lexical_relevance(self.Q, "6 7 meme compilation the best of"), 0.0)
        self.assertEqual(_lexical_relevance(self.Q, "South Dakota road project the table of"), 0.0)
        # A genuinely relevant article scores high.
        self.assertGreater(
            _lexical_relevance(self.Q, "The Spurs clinch a 2026 NBA Finals berth in San Antonio"), 0.5)

    def _pipeline(self, min_relevance):
        return NewsPipeline(use_query_planner=False, summarize_articles=False,
                            use_embeddings=False, min_relevance=min_relevance)

    def test_floor_drops_irrelevant_sources(self):
        ranked = [
            {"title": "Spurs clinch Finals berth", "relevance": 0.8, "source_channel": "web", "url": "a"},
            {"title": "6 7 meme", "relevance": 0.04, "source_channel": "web", "url": "b"},
            {"title": "SD.gov roads", "relevance": 0.0, "source_channel": "gdelt", "url": "c"},
        ]
        out = self._pipeline(0.3).select_diverse_sources(ranked, 5)
        self.assertEqual([a["url"] for a in out], ["a"])  # only the relevant one survives

    def test_all_junk_returns_no_sources(self):
        ranked = [
            {"title": "meme", "relevance": 0.05, "source_channel": "web", "url": "b"},
            {"title": "unrelated pdf", "relevance": 0.02, "source_channel": "google-news", "url": "c"},
        ]
        # Better to cite nothing (-> honest low-confidence) than ground on junk.
        self.assertEqual(self._pipeline(0.3).select_diverse_sources(ranked, 5), [])

    def test_floor_off_by_default(self):
        ranked = [{"title": "x", "relevance": 0.01, "source_channel": "web", "url": "a"}]
        self.assertEqual(len(self._pipeline(0.0).select_diverse_sources(ranked, 5)), 1)

    def test_bm25_scores_rank_relevant_higher(self):
        scores = _bm25_scores(["fed", "rate", "cut"],
                              [["fed", "rate", "cut", "decision"], ["weather", "today", "sunny"]])
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(scores[1], 0.0)

    def test_rrf_fuse_combines_rankings(self):
        # Doc 0 ranked first by both -> highest fused score.
        fused = _rrf_fuse([[0, 1, 2], [0, 2, 1]], 3)
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[0], fused[2])
        # Agreement on the loser keeps it last.
        self.assertLessEqual(fused[1], fused[0])

    def test_cross_encoder_rerank_reorders_and_annotates(self):
        # Equal dense vectors so the reranker decides the head order.
        def fake_embed(texts):
            return [[1.0, 0.0] for _ in texts]

        def fake_rerank(query, texts):
            return [1.0 if "finals" in t.lower() else 0.1 for t in texts]

        pipe = NewsPipeline(use_query_planner=False, summarize_articles=False,
                            use_embeddings=False, embed_fn=fake_embed,
                            rerank_fn=fake_rerank, min_relevance=0.0)
        ranked = pipe.rank("Who wins the NBA Finals?", [
            {"title": "A weather report for tomorrow", "url": "w"},
            {"title": "Spurs reach the NBA Finals", "url": "f"},
        ])
        self.assertEqual(ranked[0]["url"], "f")          # reranker promoted the Finals article
        self.assertEqual(ranked[0]["rerank_score"], 1.0)

    def test_rerank_failure_falls_back_to_hybrid(self):
        def fake_embed(texts):
            return [[1.0, 0.0] for _ in texts]

        def boom(query, texts):
            raise RuntimeError("reranker down")

        pipe = NewsPipeline(use_query_planner=False, summarize_articles=False,
                            use_embeddings=False, embed_fn=fake_embed, rerank_fn=boom)
        # Must not raise — falls back to the hybrid order.
        ranked = pipe.rank("NBA Finals", [{"title": "Spurs NBA Finals", "url": "a"}])
        self.assertEqual(len(ranked), 1)

    def test_semantic_ranking_via_injected_embedder(self):
        # Fake shared embedder: topical texts -> [1,0], junk -> [0,1].
        def fake_embed(texts):
            return [[1.0, 0.0] if ("spurs" in t.lower() or "finals" in t.lower())
                    else [0.0, 1.0] for t in texts]

        pipe = NewsPipeline(use_query_planner=False, summarize_articles=False,
                            use_embeddings=False, embed_fn=fake_embed, min_relevance=0.25)
        ranked = pipe.rank(self.Q, [
            {"title": "random 6 7 meme", "source_channel": "web", "url": "junk"},
            {"title": "Spurs reach the NBA Finals", "source_channel": "web", "url": "real"},
        ])
        # Semantic match puts the relevant article first with ~1.0 relevance, junk ~0.
        self.assertEqual(ranked[0]["url"], "real")
        self.assertGreater(ranked[0]["relevance"], 0.9)
        self.assertLess(ranked[1]["relevance"], 0.1)
        # And the floor then drops the junk entirely.
        kept = pipe.select_diverse_sources(ranked, 5)
        self.assertEqual([a["url"] for a in kept], ["real"])


if __name__ == "__main__":
    unittest.main()
