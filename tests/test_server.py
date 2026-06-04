from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_analytics.duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale.server import (  # noqa: E402
    _ANALYTICS_DB,
    _cache_get,
    _cache_set,
    _issue_session,
    _local_cache,
    _rate_limiter,
    _state,
    app,
)


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.response = {
            "predicted_answer": "Yes",
            "confidence": 0.7,
            "rationale": "Evidence supports a yes forecast.",
        }

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return json.dumps(self.response)


class FakeEvidencePipeline:
    def __init__(self):
        self.calls = []

    def fetch_summarize_rank(self, question, top_k=5):
        self.calls.append((question, top_k))
        return [
            {
                "title": "Central bank signals policy shift",
                "source": "Example News",
                "url": "https://example.com/rates",
                "publish_date": "2026-05-01",
                "summary": "Officials discussed conditions for a possible rate cut.",
                "relevance_score": 0.91,
                "search_query": "Federal Reserve rate cut July 2026",
            }
        ]


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.evidence_pipeline = FakeEvidencePipeline()
        _local_cache.clear()
        _rate_limiter._log.clear()  # isolate tests from cross-test rate-limit accumulation
        _state.clear()
        _state.update({
            "provider": self.provider,
            "evidence_pipeline": self.evidence_pipeline,
            "variants": {
                "variant0_neutral_baseline": SimpleNamespace(
                    output_fields=("predicted_answer", "confidence", "rationale")
                )
            },
            "system_prompt": "System",
            "prompt_templates": {
                "variant0_neutral_baseline": "[question]\nReturn JSON.",
            },
            "temperature": 0.0,
            "max_tokens": 256,
            "model_key": "test-model",
        })
        self.client = TestClient(app)

    def tearDown(self):
        _state.clear()
        _local_cache.clear()
        if _ANALYTICS_DB.exists():
            _ANALYTICS_DB.unlink()

    def test_predict_fetches_and_returns_evidence(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "description": "A forecasting question.",
                "variant": "variant0_neutral_baseline",
                "evidence_top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertEqual(payload["rationale"], "Evidence supports a yes forecast.")
        self.assertEqual(payload["model_rationale"], "Evidence supports a yes forecast.")
        self.assertEqual(payload["evidence_error"], None)
        self.assertEqual(payload["evidence_sources"][0]["source"], "Example News")
        self.assertEqual(
            payload["evidence_sources"][0]["url"],
            "https://example.com/rates",
        )
        self.assertEqual(len(payload["evidence_articles"]), 1)
        self.assertEqual(payload["evidence_articles"][0]["relevance_score"], 0.91)
        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will the Fed cut rates before July 31, 2026?", 3)],
        )
        self.assertIn("Central bank signals", self.provider.calls[0][1]["content"])

    def test_predict_uses_supplied_articles_without_fetching(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will event X happen?",
                "news_articles": [
                    {
                        "title": "Supplied evidence",
                        "summary": "The caller already provided this article.",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_articles"][0]["title"], "Supplied evidence")
        self.assertEqual(self.evidence_pipeline.calls, [])

    def test_predict_strips_html_from_returned_evidence(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will event X happen?",
                "news_articles": [
                    {
                        "title": "Supplied evidence",
                        "source": "Example News",
                        "summary": '<a href="https://example.com">Evidence</a>&nbsp;details',
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["evidence_articles"][0]["summary"], "Evidence details")

    def test_predict_returns_prediction_market_edge(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "question_type": "binary",
                "market_platform": "Polymarket",
                "market_url": "https://example.com/market",
                "market_probability": 42,
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        analysis = payload["market_analysis"]
        self.assertEqual(analysis["platform"], "Polymarket")
        self.assertEqual(analysis["market_url"], "https://example.com/market")
        self.assertEqual(analysis["outcome"], "Yes")
        self.assertAlmostEqual(analysis["market_probability"], 0.42)
        self.assertAlmostEqual(analysis["model_probability"], 0.7)
        self.assertAlmostEqual(analysis["edge"], 0.28)
        self.assertEqual(analysis["stance"], "model_above_market")
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("Prediction Market Context:", prompt)
        self.assertIn("Market-Implied Probability: 0.42", prompt)

    def test_chat_mode_uses_conversational_prompt_not_json_template(self):
        from analyzing_llm_rationale.server import _CHAT_SYSTEM_PROMPT

        response = self.client.post(
            "/predict",
            json={
                "question": "can i talk to you?",
                "chat_mode": True,
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question_type"], "chat")
        system_message = self.provider.calls[0][0]["content"]
        user_message = self.provider.calls[0][-1]["content"]
        self.assertEqual(system_message, _CHAT_SYSTEM_PROMPT)
        # No forecast/JSON typing instruction is appended in chat mode.
        self.assertNotIn("Only binary questions", user_message)
        self.assertNotIn('"type":', user_message)

    def test_markets_polymarket_endpoint_returns_quote(self):
        import analyzing_llm_rationale.market_data as md

        quote = {
            "platform": "Polymarket",
            "question": "Will X happen?",
            "market_url": "https://polymarket.com/market/will-x",
            "outcome": "Yes",
            "probability": 0.62,
            "outcomes": [
                {"label": "Yes", "probability": 0.62},
                {"label": "No", "probability": 0.38},
            ],
        }
        with mock.patch.object(md, "fetch_polymarket", lambda slug=None, market_id=None: quote):
            response = self.client.get("/markets/polymarket?slug=will-x")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["platform"], "Polymarket")
        self.assertAlmostEqual(body["probability"], 0.62)
        self.assertEqual(len(body["outcomes"]), 2)

    def test_markets_polymarket_requires_identifier(self):
        self.assertEqual(self.client.get("/markets/polymarket").status_code, 422)

    def test_markets_kalshi_not_found_maps_to_404(self):
        import analyzing_llm_rationale.market_data as md

        def boom(ticker):
            raise md.MarketDataError("Kalshi market not found.")

        with mock.patch.object(md, "fetch_kalshi", boom):
            response = self.client.get("/markets/kalshi?ticker=NOPE")
        self.assertEqual(response.status_code, 404)

    def test_agent_analyze_question_only_runs_skill(self):
        response = self.client.post(
            "/agent/analyze",
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "evidence_top_k": 3,
                "skills": [{"name": "Base rate check", "instruction": "Compare to historical base rates."}],
            },
        )
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["recommendation"], "no_market_price")  # no price supplied
        self.assertAlmostEqual(report["model_probability"], 0.7)
        self.assertTrue(report["thesis"])
        self.assertIn("forecast", report["pipeline"])
        self.assertIn("skills", report["pipeline"])
        self.assertEqual(len(report["skills"]), 1)
        self.assertEqual(report["skills"][0]["name"], "Base rate check")

    def test_agent_analyze_builtin_skills(self):
        response = self.client.post(
            "/agent/analyze",
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "evidence_top_k": 2,
                "builtin_skills": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        names = {s["name"] for s in response.json()["skills"]}
        self.assertEqual(names, {"Base rate", "Scenario decomposition", "Red team", "Key drivers"})

    def test_agent_analyze_tool_loop_runs(self):
        # FakeProvider returns forecast JSON (no action/final), so the loop treats
        # it as a final answer without calling tools — the deterministic backstop
        # must then run `forecast` itself so structured fields populate.
        response = self.client.post(
            "/agent/analyze",
            json={"question": "Will it rain tomorrow?", "tool_loop": True, "max_tool_steps": 2},
        )
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertIn("tool_loop", report["pipeline"])
        self.assertIn("forecast", report["pipeline"])           # backstop forecast ran
        self.assertAlmostEqual(report["model_probability"], 0.7)  # populated, not null

    def test_agent_analyze_fetches_market_and_recommends(self):
        import analyzing_llm_rationale.market_data as md

        quote = {
            "platform": "Polymarket",
            "question": "Will the Fed cut rates before September 30, 2026?",
            "market_url": "https://polymarket.com/market/fed",
            "outcome": "Yes",
            "probability": 0.40,
            "outcomes": [{"label": "Yes", "probability": 0.40}, {"label": "No", "probability": 0.60}],
        }
        with mock.patch.object(md, "fetch_polymarket", lambda slug=None, market_id=None: quote):
            response = self.client.post(
                "/agent/analyze",
                json={"platform": "polymarket", "slug": "fed", "evidence_top_k": 2},
            )
        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["platform"], "Polymarket")
        self.assertIn("resolve_market", report["pipeline"])
        self.assertAlmostEqual(report["market_probability"], 0.40)
        self.assertAlmostEqual(report["model_probability"], 0.70)
        self.assertAlmostEqual(report["edge"], 0.30)
        self.assertEqual(report["recommendation"], "buy_yes")

    def test_agent_analyze_requires_question_or_market(self):
        response = self.client.post("/agent/analyze", json={"evidence_top_k": 3})
        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_short_standalone_question(self):
        response = self.client.post("/predict", json={"question": "why?", "attach_evidence": False})
        self.assertEqual(response.status_code, 422)

    def test_predict_allows_short_followup_with_history(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "why?",
                "attach_evidence": False,
                "history": [
                    {"role": "user", "content": "Will the Fed cut rates before 2027?"},
                    {"role": "assistant", "content": "Probably yes."},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_agent_analyze_passes_history_for_followups(self):
        response = self.client.post(
            "/agent/analyze",
            json={
                "question": "what about by June instead?",
                "attach_evidence": False,
                "evidence_top_k": 2,
                "history": [
                    {"role": "user", "content": "Will the Fed cut rates before September 30, 2026?"},
                    {"role": "assistant", "content": "I'd put Yes around 62%."},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        # The prior turn reached the model, giving the follow-up context.
        all_content = " ".join(m["content"] for m in self.provider.calls[0])
        self.assertIn("September 30, 2026", all_content)

    def test_agent_scan_ranks_mispriced_markets(self):
        import analyzing_llm_rationale.market_data as md

        # Model forecasts Yes 0.7 (FakeProvider). Market A at 0.40 -> edge 0.30
        # (surfaces); market B at 0.66 -> edge 0.04 (below min_edge, filtered).
        markets = [
            {"platform": "Polymarket", "question": "Will event A happen by 2027?", "market_url": "https://p/a",
             "outcome": "Yes", "probability": 0.40, "outcomes": []},
            {"platform": "Polymarket", "question": "Will event B happen by 2027?", "market_url": "https://p/b",
             "outcome": "Yes", "probability": 0.66, "outcomes": []},
        ]
        with mock.patch.object(md, "list_polymarket", lambda limit=5, query=None: markets):
            response = self.client.get("/agent/scan?platform=polymarket&limit=2&min_edge=0.1&evidence_top_k=2")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["platform"], "Polymarket")
        self.assertEqual(body["scanned"], 2)
        self.assertEqual(len(body["opportunities"]), 1)
        opp = body["opportunities"][0]
        self.assertEqual(opp["question"], "Will event A happen by 2027?")
        self.assertAlmostEqual(opp["edge"], 0.30)
        self.assertEqual(opp["recommendation"], "buy_yes")

    def test_agent_scan_both_venues_and_keyword(self):
        import analyzing_llm_rationale.market_data as md

        poly = [{"platform": "Polymarket", "question": "Will the Lakers win the NBA title by 2027?",
                 "market_url": "https://p/nba", "outcome": "Yes", "probability": 0.40, "outcomes": []}]
        kalshi = [{"platform": "Kalshi", "question": "Will an NBA team relocate by 2027?",
                   "market_url": "https://k/nba", "outcome": "Yes", "probability": 0.42, "outcomes": []}]
        seen = {}

        def fake_poly(limit=5, query=None):
            seen["poly_query"] = query
            return poly

        def fake_kalshi(limit=5, query=None):
            seen["kalshi_query"] = query
            return kalshi

        with mock.patch.object(md, "list_polymarket", fake_poly), \
             mock.patch.object(md, "list_kalshi", fake_kalshi):
            response = self.client.get("/agent/scan?platform=all&query=nba&limit=2&min_edge=0.1&evidence_top_k=2")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["platform"], "Polymarket + Kalshi")
        self.assertEqual(body["scanned"], 2)  # one from each venue
        self.assertEqual(seen["poly_query"], "nba")
        self.assertEqual(seen["kalshi_query"], "nba")

    def test_agent_scan_empty_when_keyword_matches_nothing(self):
        import analyzing_llm_rationale.market_data as md

        with mock.patch.object(md, "list_polymarket", lambda limit=5, query=None: []):
            response = self.client.get("/agent/scan?platform=polymarket&query=zzzznope")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["scanned"], 0)
        self.assertEqual(body["opportunities"], [])

    def test_agent_scan_rejects_unknown_platform(self):
        response = self.client.get("/agent/scan?platform=betfair")
        self.assertEqual(response.status_code, 422)

    def test_robots_txt_welcomes_ai_crawlers(self):
        r = self.client.get("/robots.txt")
        self.assertEqual(r.status_code, 200)
        self.assertIn("GPTBot", r.text)
        self.assertIn("ClaudeBot", r.text)
        self.assertIn("PerplexityBot", r.text)
        self.assertIn("Sitemap: https://foresea.ink/sitemap.xml", r.text)

    def test_llms_txt_describes_api(self):
        r = self.client.get("/llms.txt")
        self.assertEqual(r.status_code, 200)
        self.assertIn("# Foresea", r.text)
        self.assertIn("/predict", r.text)
        self.assertIn("/openapi.json", r.text)

    def test_sitemap_xml(self):
        r = self.client.get("/sitemap.xml")
        self.assertEqual(r.status_code, 200)
        self.assertIn("application/xml", r.headers.get("content-type", ""))
        self.assertIn("https://foresea.ink/", r.text)

    def test_openapi_spec_is_public(self):
        # Agents introspect the API via the OpenAPI spec.
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["info"]["title"], "Foresea Intelligence API")

    def test_track_record_tick_disabled_without_token(self):
        import analyzing_llm_rationale.server as srv
        with mock.patch.object(srv, "_TRACK_RECORD_TOKEN", None):
            r = self.client.post("/track-record/tick")
        self.assertEqual(r.status_code, 503)

    def test_track_record_tick_rejects_bad_token(self):
        import analyzing_llm_rationale.server as srv
        with mock.patch.object(srv, "_TRACK_RECORD_TOKEN", "secret-token"):
            r = self.client.post("/track-record/tick", headers={"X-Track-Token": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_news_articles_tolerates_invalid_fields(self):
        from analyzing_llm_rationale.server import _news_articles
        # A negative relevance_score (from negative cosine similarity) must not
        # 500 the response — it's repaired/skipped, not raised.
        articles = _news_articles([
            {"title": "Good", "url": "https://x.com", "source": "X", "relevance_score": 0.5},
            {"title": "Bad", "url": "https://y.com", "source": "Y", "relevance_score": -0.3},
        ])
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[1].title, "Bad")

    def test_predict_includes_knowledge_base_for_signed_in_user(self):
        from analyzing_llm_rationale import rag

        class FakeEmbedder:
            def encode(self, texts, normalize_embeddings=True):
                import math
                import re
                vocab = ["fed", "rate", "cut", "cpi", "spacex"]
                out = []
                for t in texts:
                    toks = set(re.findall(r"[a-z]+", t.lower()))
                    v = [1.0 if w in toks else 0.0 for w in vocab]
                    n = math.sqrt(sum(x * x for x in v)) or 1.0
                    out.append([x / n for x in v])
                return out

        rag.set_embedder(FakeEmbedder())
        try:
            headers = {"Authorization": "Bearer " + _issue_session("kbuser", "k@e.com", "K", "")}
            ingest = self.client.post(
                "/rag/ingest",
                json={"text": "The Fed will cut the rate soon.", "title": "My note"},
                headers=headers,
            )
            self.assertEqual(ingest.status_code, 200)

            response = self.client.post(
                "/predict",
                json={"question": "Will the Fed cut the rate this year?", "attach_evidence": False},
                headers=headers,
            )
            self.assertEqual(response.status_code, 200)
            sources = [s["source"] for s in response.json().get("evidence_sources", [])]
            self.assertIn("Knowledge base", sources)
        finally:
            rag.set_embedder(None)

    def test_records_anonymous_page_visit(self):
        response = self.client.post(
            "/analytics/visit",
            json={
                "path": "/",
                "referrer": "",
                "timezone": "Europe/Berlin",
            },
            headers={"user-agent": "test-client"},
        )
        self.assertEqual(response.status_code, 200)

        summary = self.client.get("/analytics/summary")
        self.assertEqual(summary.status_code, 200)
        payload = summary.json()
        self.assertGreaterEqual(payload["total_visits"], 1)
        self.assertGreaterEqual(payload["unique_visitors"], 1)
        self.assertEqual(payload["by_day"][0]["visits"], 1)

    def test_predict_multiple_choice_returns_options(self):
        self.provider.response = {
            "type": "multiple_choice",
            "options": [
                {"label": "Alice", "probability": 0.2},
                {"label": "Bob", "probability": 0.7},
                {"label": "Carol", "probability": 0.1},
            ],
            "rationale": "Bob has the strongest polling and fundraising evidence.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "Who will win the Example City mayoral election?",
                "question_type": "multiple_choice",
                "options": ["Alice", "Bob", "Carol"],
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "multiple_choice")
        self.assertEqual(payload["predicted_answer"], "Bob")
        self.assertEqual(payload["confidence"], 0.7)
        self.assertEqual(len(payload["options"]), 3)
        self.assertIn("multiple_choice", self.provider.calls[0][-1]["content"])
        self.assertIn("Alice, Bob, Carol", self.provider.calls[0][-1]["content"])
        self.assertIn("overrides any earlier variant template", self.provider.calls[0][-1]["content"])
        self.assertIn("Only binary questions should use a Yes/No", self.provider.calls[0][-1]["content"])

    def test_predict_numeric_returns_range_forecast(self):
        self.provider.response = {
            "type": "numeric",
            "p10": 42,
            "p50": 55,
            "p90": 73,
            "unit": "USD",
            "rationale": "Recent guidance supports a mid-range estimate.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "What will Example Corp revenue be in Q4 2026?",
                "question_type": "numeric",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "numeric")
        self.assertEqual(payload["predicted_answer"], "55")
        self.assertEqual(payload["confidence"], None)
        self.assertEqual(
            payload["range_forecast"],
            {"p10": "42", "p50": "55", "p90": "73", "unit": "USD"},
        )
        self.assertIn('"type":"numeric"', self.provider.calls[0][-1]["content"])
        self.assertIn("Only binary questions should use a Yes/No", self.provider.calls[0][-1]["content"])

    def test_predict_without_question_type_asks_model_to_infer_schema(self):
        self.provider.response = {
            "type": "numeric",
            "p10": 2.1,
            "p50": 2.6,
            "p90": 3.4,
            "unit": "%",
            "rationale": "Inflation is likely to stay near recent targets.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "What will US CPI inflation be in December 2026?",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "numeric")
        self.assertEqual(payload["predicted_answer"], "2.6")
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("First infer the question type", prompt)
        self.assertIn("- numeric:", prompt)

    def test_predict_date_returns_range_forecast(self):
        self.provider.response = {
            "type": "date",
            "p10": "2026-07-01",
            "p50": "2026-09-15",
            "p90": "2026-12-31",
            "rationale": "The event is most likely in the second half of 2026.",
        }

        response = self.client.post(
            "/predict",
            json={
                "question": "When will the Example spacecraft launch?",
                "question_type": "date",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "date")
        self.assertEqual(payload["predicted_answer"], "2026-09-15")
        self.assertEqual(payload["range_forecast"]["p50"], "2026-09-15")

    def test_predict_rejects_unknown_question_type(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "What will Example Corp revenue be in Q4 2026?",
                "question_type": "essay",
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_authenticated_user_can_sync_chat_conversations(self):
        token = _issue_session("user-123", "user@example.com", "User Example", "")
        headers = {"Authorization": f"Bearer {token}"}
        conversation = {
            "id": "conv_test",
            "title": "Fed forecast",
            "createdAt": 1000,
            "updatedAt": 2000,
            "messages": [
                {
                    "id": "msg_user",
                    "role": "user",
                    "content": "Will the Fed cut rates before July 2026?",
                    "createdAt": 1001,
                },
                {
                    "id": "msg_assistant",
                    "role": "assistant",
                    "content": "Probably yes.",
                    "createdAt": 1002,
                },
            ],
        }

        save_response = self.client.put(
            "/chat/conversations/conv_test",
            json=conversation,
            headers=headers,
        )
        self.assertEqual(save_response.status_code, 200)

        list_response = self.client.get("/chat/conversations", headers=headers)
        self.assertEqual(list_response.status_code, 200)
        saved = list_response.json()["conversations"][0]
        self.assertEqual(saved["id"], "conv_test")
        self.assertEqual([m["id"] for m in saved["messages"]], ["msg_user", "msg_assistant"])

        delete_response = self.client.delete("/chat/conversations/conv_test", headers=headers)
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(self.client.get("/chat/conversations", headers=headers).json()["conversations"], [])

    def test_chat_conversation_sync_requires_session(self):
        response = self.client.get("/chat/conversations")
        self.assertEqual(response.status_code, 401)

    def test_github_auth_unconfigured_returns_503(self):
        # GITHUB_CLIENT_ID/SECRET are unset in tests.
        response = self.client.post("/auth/github", json={"code": "abc"})
        self.assertEqual(response.status_code, 503)

    def test_github_auth_issues_session(self):
        import analyzing_llm_rationale.server as srv

        profile = {
            "sub": "github:123",
            "email": "octo@example.com",
            "name": "Octo Cat",
            "picture": "https://avatars.example/octo.png",
        }
        with mock.patch.object(srv, "_exchange_github_code", lambda code, redirect_uri: profile):
            response = self.client.post(
                "/auth/github", json={"code": "abc", "redirect_uri": "https://foresea.ink/"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user_id"], "github:123")
        self.assertEqual(body["email"], "octo@example.com")
        self.assertTrue(body["token"])

    def test_auth_config_exposes_github_client_id_field(self):
        cfg = self.client.get("/auth/config").json()
        self.assertIn("github_client_id", cfg)
        self.assertIn("google_client_id", cfg)

    def test_register_then_login_with_email_password(self):
        register = self.client.post(
            "/auth/register",
            json={"email": "Trader@Example.com", "password": "supersecret1", "name": "Ada"},
        )
        self.assertEqual(register.status_code, 200)
        body = register.json()
        self.assertEqual(body["email"], "trader@example.com")
        self.assertEqual(body["user_id"], "trader@example.com")
        self.assertEqual(body["name"], "Ada")
        self.assertTrue(body["token"])

        # The token authenticates subsequent requests.
        me = self.client.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["email"], "trader@example.com")

        # Login with the same credentials (email is case-insensitive).
        login = self.client.post(
            "/auth/login",
            json={"email": "trader@example.com", "password": "supersecret1"},
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(login.json()["user_id"], "trader@example.com")

    def test_register_rejects_duplicate_email(self):
        payload = {"email": "dup@example.com", "password": "supersecret1"}
        self.assertEqual(self.client.post("/auth/register", json=payload).status_code, 200)
        second = self.client.post("/auth/register", json=payload)
        self.assertEqual(second.status_code, 409)

    def test_register_rejects_short_password(self):
        response = self.client.post(
            "/auth/register",
            json={"email": "weak@example.com", "password": "short"},
        )
        self.assertEqual(response.status_code, 422)

    def test_register_rejects_invalid_email(self):
        response = self.client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "supersecret1"},
        )
        self.assertEqual(response.status_code, 422)

    def test_login_with_wrong_password_is_unauthorized(self):
        self.client.post(
            "/auth/register",
            json={"email": "real@example.com", "password": "supersecret1"},
        )
        response = self.client.post(
            "/auth/login",
            json={"email": "real@example.com", "password": "wrongpassword"},
        )
        self.assertEqual(response.status_code, 401)

    def test_login_unknown_email_is_unauthorized(self):
        response = self.client.post(
            "/auth/login",
            json={"email": "ghost@example.com", "password": "supersecret1"},
        )
        self.assertEqual(response.status_code, 401)

    def test_identical_predict_requests_are_cached(self):
        payload = {
            "question": "Will the Fed cut rates before December 31, 2026?",
            "question_type": "binary",
            "attach_evidence": False,
        }
        first = self.client.post("/predict", json=payload)
        second = self.client.post("/predict", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        # The model provider is invoked only once; the second call hits the cache.
        self.assertEqual(len(self.provider.calls), 1)

    def test_short_followup_skips_fresh_evidence(self):
        # A short follow-up in a thread should be answered from context, not a
        # fresh literal web search (which derails on odd queries).
        self.client.post(
            "/predict",
            json={
                "question": "WE is 90+",
                "attach_evidence": True,
                "history": [
                    {"role": "user", "content": "Who wins AL vs WE in the LPL series?"},
                    {"role": "assistant", "content": "AL looks favoured ~60%."},
                ],
            },
        )
        self.assertEqual(self.evidence_pipeline.calls, [])  # no fresh retrieval

    def test_substantive_question_with_history_still_retrieves(self):
        self.client.post(
            "/predict",
            json={
                "question": "Will the Federal Reserve cut interest rates before September 2026?",
                "attach_evidence": True,
                "evidence_top_k": 3,
                "history": [{"role": "user", "content": "earlier turn"}],
            },
        )
        self.assertEqual(len(self.evidence_pipeline.calls), 1)

    def test_parse_market_url(self):
        from analyzing_llm_rationale.server import _parse_market_url
        self.assertEqual(
            _parse_market_url("https://polymarket.com/esports/lpl/lol-al-we-2026-06-01"),
            ("polymarket", "slug", "lol-al-we-2026-06-01"),
        )
        self.assertEqual(
            _parse_market_url("check https://kalshi.com/markets/KXFED-26SEP now"),
            ("kalshi", "ticker", "KXFED-26SEP"),
        )
        self.assertIsNone(_parse_market_url("just text"))

    def test_predict_with_history_is_not_cached(self):
        payload = {
            "question": "Will the Fed cut rates before December 31, 2026?",
            "question_type": "binary",
            "attach_evidence": False,
            "history": [{"role": "user", "content": "earlier turn"}],
        }
        self.client.post("/predict", json=payload)
        self.client.post("/predict", json=payload)
        self.assertEqual(len(self.provider.calls), 2)

    def test_evidence_retrieval_is_cached_across_requests(self):
        payload = {
            "question": "Will the Fed cut rates before July 31, 2026?",
            "variant": "variant0_neutral_baseline",
            "evidence_top_k": 3,
            "history": [{"role": "user", "content": "x"}],  # disable full-response cache
        }
        self.client.post("/predict", json=payload)
        self.client.post("/predict", json=payload)
        # Evidence pipeline fetched once even though the model ran twice.
        self.assertEqual(len(self.evidence_pipeline.calls), 1)
        self.assertEqual(len(self.provider.calls), 2)

    def test_cache_get_set_roundtrip_and_ttl(self):
        _cache_set("unit-key", {"value": 7}, ttl=60)
        self.assertEqual(_cache_get("unit-key"), {"value": 7})
        self.assertIsNone(_cache_get("missing-key"))
        _cache_set("zero-ttl", {"v": 1}, ttl=0)  # ttl<=0 is a no-op
        self.assertIsNone(_cache_get("zero-ttl"))

    def test_custom_provider_base_url_routes_to_own_endpoint(self):
        import analyzing_llm_rationale.providers as providers_mod

        captured = {}

        class FakeCustomProvider:
            def __init__(self, model_name, api_key, base_url):
                captured.update(model=model_name, key=api_key, base_url=base_url)

            def chat_completion(self, messages, temperature, max_tokens):
                return json.dumps(
                    {"predicted_answer": "Yes", "confidence": 0.6, "rationale": "ok"}
                )

        with mock.patch.object(providers_mod, "OpenAICompatibleProvider", FakeCustomProvider):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will X happen by 2027?",
                    "question_type": "binary",
                    "attach_evidence": False,
                    "openrouter_api_key": "sk-test",
                    "openrouter_model": "gpt-4o",
                    "provider_base_url": "https://api.openai.com/v1/chat/completions",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["base_url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["model"], "gpt-4o")
        self.assertEqual(captured["key"], "sk-test")
        # The server's default provider must not be called for a BYOK request.
        self.assertEqual(self.provider.calls, [])

    def test_custom_provider_rejects_unsafe_base_url(self):
        for bad_url in [
            "http://api.openai.com/v1/chat/completions",  # not https
            "https://localhost/v1/chat/completions",
            "https://169.254.169.254/latest/meta-data",  # cloud metadata
            "https://10.0.0.5/v1/chat/completions",       # private range
            "https://metadata.google.internal/x",
        ]:
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will X happen by 2027?",
                    "attach_evidence": False,
                    "openrouter_api_key": "sk-test",
                    "openrouter_model": "gpt-4o",
                    "provider_base_url": bad_url,
                },
            )
            self.assertEqual(response.status_code, 422, bad_url)


if __name__ == "__main__":
    unittest.main()
