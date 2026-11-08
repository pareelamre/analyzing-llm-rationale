from __future__ import annotations

import json
import unittest
from threading import Event
from unittest import mock

from analyzing_llm_rationale import agent_research as research


class ResearchPacketTests(unittest.TestCase):
    def articles(self, pipeline, articles):
        pipeline.return_value.fetch.return_value = articles
        pipeline.return_value.rank.side_effect = lambda query, rows: rows
        pipeline.return_value.select_diverse_sources.side_effect = lambda rows, top_k: rows

    def quote(self, ident="KXTEST"):
        return {"platform": "Kalshi", "ident": ident, "question": "Will the event happen?",
                "yes_bid": 0.4, "yes_ask": 0.45, "api_key": "private-value"}

    def test_packet_contains_bounded_books_and_keyless_sources(self):
        articles = [{"title": "Official update", "url": "https://example.test/update",
                     "summary": "Dated evidence.", "publish_date": "2026-10-31",
                     "source_channel": "google-news", "relevance": 0.8}]
        with (
            mock.patch.object(research, "NewsPipeline") as pipeline,
            mock.patch.object(research.market_data, "fetch_kalshi_orderbook",
                              return_value={"yes": [[30, 10], [41, 20], [40, 30], [39, 40]],
                                            "no": [[54, 10]], "private": "do-not-copy"}),
        ):
            self.articles(pipeline, articles)
            steps = research.prefetch_market_research([self.quote()])
        pipeline.assert_called_once_with(use_query_planner=False, summarize_articles=False,
                                         use_embeddings=False, fetch_sources=research.NEWS_SOURCES,
                                         min_relevance=research.MIN_RELEVANCE)
        packet = json.loads(steps[0]["observation"])
        self.assertEqual(packet["orderbook"]["yes"][0], [41, 20])
        self.assertEqual(len(packet["orderbook"]["yes"]), 3)
        self.assertEqual(packet["buy_yes_ask_dollars"], 0.46)
        self.assertEqual(packet["buy_no_ask_dollars"], 0.59)
        self.assertIn("cents", packet["orderbook_convention"])
        self.assertEqual(packet["sources"][0]["url"], "https://example.test/update")
        self.assertEqual(packet["sources"][0]["published_at"], "2026-10-31")
        self.assertNotIn("private-value", steps[0]["observation"])
        self.assertNotIn("do-not-copy", steps[0]["observation"])
        self.assertEqual(steps[0]["initiated_by"], "system_prefetch")

    def test_fanout_is_capped_deduplicated_and_failures_are_explicit(self):
        quotes = [self.quote(), self.quote(), self.quote("KXSECOND"), self.quote("KXTHIRD")]
        with (
            mock.patch.object(research, "NewsPipeline", side_effect=RuntimeError("private error")),
            mock.patch.object(research.market_data, "fetch_kalshi_orderbook",
                              side_effect=RuntimeError("private error")) as books,
        ):
            steps = research.prefetch_market_research(quotes)
        self.assertEqual(len(steps), 2)
        self.assertEqual(books.call_count, 2)
        for step in steps:
            packet = json.loads(step["observation"])
            self.assertEqual(packet["evidence_status"], "unavailable")
            self.assertEqual(packet["orderbook_status"], "unavailable")
            self.assertEqual(packet["sources"], [])
            self.assertNotIn("private error", step["observation"])

    def test_generic_headlines_and_unsafe_urls_are_not_evidence(self):
        articles = [{"title": "Generic feed", "url": "https://example.test/feed",
                     "source_channel": "stooq"},
                    {"title": "Unsafe", "url": "javascript:alert(1)"},
                    {"title": "Unrelated", "url": "https://example.test/other", "relevance": 0.01}]
        with (mock.patch.object(research, "NewsPipeline") as pipeline,
              mock.patch.object(research.market_data, "fetch_kalshi_orderbook", return_value={})):
            self.articles(pipeline, articles)
            packet = json.loads(research.prefetch_market_research([self.quote()])[0]["observation"])
        self.assertEqual(packet["sources"], [])
        self.assertEqual(packet["evidence_status"], "empty")
        self.assertEqual(packet["orderbook_status"], "empty")

    def test_polymarket_uses_the_yes_token_not_slug_as_a_kalshi_ticker(self):
        quote = research.market_data._polymarket_quote({
            "slug": "a-market-slug", "question": "Will the event happen?",
            "outcomes": ["No", "Yes"], "outcomePrices": ["0.4", "0.6"],
            "clobTokenIds": ["54321", "12345"],
        })
        with (mock.patch.object(research, "NewsPipeline") as pipeline,
              mock.patch.object(research.market_data, "fetch_polymarket_orderbook",
                                return_value={"asks": [{"price": "0.8", "size": "10"},
                                                       {"price": "0.6", "size": "20"}]}) as books,
              mock.patch.object(research.market_data, "fetch_kalshi_orderbook") as kalshi):
            self.articles(pipeline, [])
            packet = json.loads(research.prefetch_market_research([quote])[0]["observation"])
        books.assert_called_once_with("12345")
        kalshi.assert_not_called()
        self.assertEqual(packet["orderbook"]["asks"][0]["price"], "0.6")

    def test_invalid_identity_is_skipped_before_network_access(self):
        with (mock.patch.object(research, "NewsPipeline") as pipeline,
              mock.patch.object(research.market_data, "fetch_kalshi_orderbook") as books):
            self.assertEqual(research.prefetch_market_research([None, "bad", self.quote("../../private")]), [])
        pipeline.assert_not_called()
        books.assert_not_called()

    def test_large_external_fields_and_malformed_levels_are_bounded(self):
        quote = self.quote()
        quote.update(question="Q" * 10000, resolution_criteria="R" * 10000)
        articles = [{"title": "T" * 10000, "url": "https://example.test/update",
                     "summary": "S" * 10000, "relevance": 0.8} for _ in range(10)]
        with (mock.patch.object(research, "NewsPipeline") as pipeline,
              mock.patch.object(research.market_data, "fetch_kalshi_orderbook", return_value={
                  "yes": [[float("nan"), 10], [40, -1], {}, ["bad", 2], [41, 3]],
              })):
            self.articles(pipeline, articles)
            step = research.prefetch_market_research([quote])[0]
        packet = json.loads(step["observation"])
        self.assertEqual(packet["orderbook"]["yes"], [[41, 3]])
        self.assertEqual(len(packet["sources"]), 2)
        self.assertLess(len(step["observation"]), 4000)

    def test_one_malformed_article_does_not_discard_good_sources(self):
        articles = [{"url": "http://[::1", "relevance": 0.8},
                    {"url": "https://example.test/bad", "relevance": None},
                    {"url": "https://example.test/nan", "relevance": float("nan")},
                    {"url": "https://example.test/good", "relevance": 0.8}]
        with mock.patch.object(research, "NewsPipeline") as pipeline:
            self.articles(pipeline, articles)
            evidence = research.search_market_evidence("event")
        self.assertEqual(evidence["evidence_status"], "available")
        self.assertEqual([s["url"] for s in evidence["sources"]], ["https://example.test/good"])

    def test_real_news_ranking_and_kalshi_adapter_contract_without_inference(self):
        article = {"title": "Will the event happen? Event happens tomorrow", "summary": "The event will happen.",
                   "url": "https://example.test/good", "source_channel": "google-news"}
        with (mock.patch.object(research.NewsPipeline, "fetch", return_value=[article]),
              mock.patch.object(research.NewsPipeline, "summarize", side_effect=AssertionError("No inference")),
              mock.patch.object(research.market_data, "_get_json", return_value={
                  "orderbook": {"yes_dollars": [["0.41", "3"]], "no_dollars": [["0.54", "4"]]}})):
            step = research.prefetch_market_research([self.quote()])[0]
        packet = json.loads(step["observation"])
        self.assertTrue(step["research_available"])
        self.assertEqual(packet["evidence_status"], "available")
        self.assertEqual(packet["orderbook_status"], "available")
        self.assertEqual(packet["buy_yes_ask_dollars"], 0.46)

    def test_deadline_returns_missing_data_without_waiting_for_a_slow_worker(self):
        release, started, finished = Event(), Event(), Event()
        def stalled(quote):
            started.set()
            release.wait(timeout=2)
            finished.set()
            return research._unavailable_packet(quote)
        try:
            with (mock.patch.object(research, "PREFETCH_TIMEOUT_S", 0.02),
                  mock.patch.object(research, "_market_packet", side_effect=stalled)):
                steps = research.prefetch_market_research([self.quote()])
            self.assertTrue(started.is_set())
            self.assertFalse(finished.is_set())
            self.assertFalse(steps[0]["research_available"])
            self.assertEqual(json.loads(steps[0]["observation"])["evidence_status"], "unavailable")
        finally:
            release.set()
            self.assertTrue(finished.wait(timeout=2))
