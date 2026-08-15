from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_analytics.duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server as server_module  # noqa: E402
from analyzing_llm_rationale import trading as trading_module  # noqa: E402
from analyzing_llm_rationale.providers import RetryableProviderError  # noqa: E402
from analyzing_llm_rationale.server import (  # noqa: E402
    _cache_get,
    _cache_set,
    _issue_session,
    _local_cache,
    _predict_rate_limiter,
    _rate_limiter,
    _state,
    app,
)


class FakeProvider:
    def __init__(self):
        self.model_name = "fake-model"
        self.last_response_model = "fake-model"
        self.calls = []
        self.max_tokens = []
        self.response = {
            "predicted_answer": "Yes",
            "confidence": 0.7,
            "rationale": "Evidence supports a yes forecast.",
        }
        self.stream_response = "Streaming answer."

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response)

    def stream_chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        mid = max(1, len(self.stream_response) // 2)
        yield self.stream_response[:mid]
        yield self.stream_response[mid:]


class FakeTradingKms:
    """In-memory KMS double that verifies the encrypted DEK's binding context."""

    def __init__(self):
        self.encrypt_requests = []
        self.decrypt_requests = []
        self._wrapped = {}

    def encrypt(self, *, request):
        self.encrypt_requests.append(request)
        ciphertext = os.urandom(16)
        self._wrapped[ciphertext] = (
            request["name"],
            request["plaintext"],
            request["additional_authenticated_data"],
        )
        return SimpleNamespace(
            ciphertext=ciphertext,
            name=f"{request['name']}/cryptoKeyVersions/7",
        )

    def decrypt(self, *, request):
        self.decrypt_requests.append(request)
        name, plaintext, aad = self._wrapped[request["ciphertext"]]
        if request["name"] != name or request["additional_authenticated_data"] != aad:
            raise ValueError("KMS authenticated data did not match")
        return SimpleNamespace(plaintext=plaintext)


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


class FailingProvider:
    def __init__(self, model_name="failing-model"):
        self.model_name = model_name
        self.calls = 0

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls += 1
        raise RetryableProviderError("upstream unavailable")


class SlowStreamProvider:
    model_name = "slow-stream-model"

    def __init__(self):
        self.calls = 0

    def stream_chat_completion(self, messages, temperature, max_tokens):
        self.calls += 1
        time.sleep(0.05)
        yield "late primary"


class EmptyStreamProvider(FakeProvider):
    def stream_chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        self.max_tokens.append(max_tokens)
        if False:
            yield ""


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.provider = FakeProvider()
        self.evidence_pipeline = FakeEvidencePipeline()
        self.analytics_db = Path(tempfile.gettempdir()) / "foresea_test_analytics.duckdb"
        server_module._ANALYTICS_DB = self.analytics_db
        if self.analytics_db.exists():
            self.analytics_db.unlink()
        self._datastore_patch = mock.patch.object(server_module, "_get_datastore", return_value=None)
        self._datastore_patch.start()
        def fake_require_auth(request):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                try:
                    return server_module._decode_session(auth[7:])
                except Exception:
                    pass
            return {"sub": "test-user", "email": "test@example.com", "name": "Test User"}

        self._require_auth_patch = mock.patch.object(
            server_module,
            "_require_auth",
            side_effect=fake_require_auth
        )
        self._require_auth_mock = self._require_auth_patch.start()
        _local_cache.clear()
        _rate_limiter._log.clear()
        _predict_rate_limiter._log.clear()
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
        self._require_auth_patch.stop()
        self._datastore_patch.stop()
        _state.clear()
        _local_cache.clear()
        if self.analytics_db.exists():
            self.analytics_db.unlink()

    def _page_context(self, response):
        marker = '<script type="application/json" id="foresea-page-context">'
        start = response.text.index(marker) + len(marker)
        end = response.text.index("</script>", start)
        return json.loads(response.text[start:end])

    def test_predict_fetches_and_returns_evidence(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "description": "A forecasting question.",
                "variant": "variant0_neutral_baseline",
                "evidence_top_k": 3,
                "chat_mode": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertTrue(payload["rationale"].startswith("Evidence supports a yes forecast."))
        self.assertIn("**Sources provided to the forecast**", payload["rationale"])
        self.assertIn("- **Example News**: Central bank signals policy shift", payload["rationale"])
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

    def test_chat_predict_appends_named_source_attribution(self):
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "variant": "variant0_neutral_baseline",
                "evidence_top_k": 3,
                "chat_mode": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        rationale = response.json()["rationale"]
        self.assertIn("**Sources provided to the forecast**", rationale)
        self.assertIn("**Example News**", rationale)
        self.assertIn("Central bank signals policy shift", rationale)
        messages = self.provider.calls[0]
        self.assertIn("Never refer to evidence only as", messages[0]["content"])
        self.assertIn(
            "Evidence 1 — Example News: Central bank signals policy shift",
            messages[1]["content"],
        )

        slow_pipeline = mock.Mock()
        def slow_fetch(question, top_k=5):
            time.sleep(0.05)
            return [{
                "title": "Late evidence",
                "source": "Slow News",
                "summary": "This arrived too late for the forecast path.",
            }]

        slow_pipeline.fetch_summarize_rank.side_effect = slow_fetch
        _state["evidence_pipeline"] = slow_pipeline

        with mock.patch.object(server_module, "_EVIDENCE_TIMEOUT_S", 0.001):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "variant": "variant0_neutral_baseline",
                    "chat_mode": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertEqual(payload["evidence_sources"], [])
        self.assertIn("Evidence retrieval timed out", payload["evidence_error"])
        self.assertEqual(slow_pipeline.fetch_summarize_rank.call_count, 1)

    def test_chat_predict_returns_probability_score(self):
        self.provider.response = "I estimate a 65% chance.\n[p:0.65]"
        response = self.client.post(
            "/predict",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "variant": "variant0_neutral_baseline",
                "attach_evidence": False,
                "chat_mode": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question_type"], "chat")
        self.assertEqual(payload["model_probability"], 0.65)
        self.assertNotIn("[p:0.65]", payload["rationale"])
        self.assertTrue(payload["rationale"].startswith("**Forecast: 65%**"))
        self.assertIn("marker is hidden from the user", self.provider.calls[0][0]["content"])

    def test_predict_skips_evidence_when_fetch_pool_is_busy(self):
        busy_slots = server_module.threading.BoundedSemaphore(1)
        self.assertTrue(busy_slots.acquire(blocking=False))
        with mock.patch.object(server_module, "_evidence_fetch_slots", busy_slots):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "variant": "variant0_neutral_baseline",
                    "chat_mode": False,
                },
            )
        busy_slots.release()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertEqual(payload["evidence_sources"], [])
        self.assertIn("Evidence retrieval is busy", payload["evidence_error"])
        self.assertEqual(self.evidence_pipeline.calls, [])

    def test_predict_marks_empty_evidence_and_does_not_cache_the_miss(self):
        empty_pipeline = mock.Mock()
        empty_pipeline.fetch_summarize_rank.return_value = []
        _state["evidence_pipeline"] = empty_pipeline
        request = {
            "question": "Will any member of Trump's Cabinet leave before August 2026?",
            "variant": "variant0_neutral_baseline",
            "evidence_top_k": 5,
            "chat_mode": True,
        }

        with mock.patch.object(server_module, "_PREDICT_CACHE_TTL", 0):
            first = self.client.post("/predict", json=request)
            second = self.client.post("/predict", json=request)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        payload = first.json()
        self.assertEqual(payload["evidence_sources"], [])
        self.assertEqual(
            payload["evidence_error"],
            "No relevant live evidence sources were found after retrying retrieval.",
        )
        self.assertEqual(empty_pipeline.fetch_summarize_rank.call_count, 2)
        system_message = self.provider.calls[0][0]["content"]
        self.assertIn("Evidence status: no relevant live sources were retrieved", system_message)
        self.assertIn("This is a retrieval failure, not evidence about the event", system_message)
        self.assertIn('Do not say or imply "no current reporting"', system_message)
        self.assertIn("label supplied pricing as **Market context**", system_message)

    def test_run_app_host_redirects_by_default(self):
        response = self.client.get(
            "/health",
            headers={"host": "analyzing-llm-rationale-hy7gvnvt4a-uc.a.run.app"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 301)
        self.assertEqual(response.headers["location"], "https://foresea.ink/health")

    def test_run_app_host_does_not_redirect_in_staging(self):
        with mock.patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=False):
            response = self.client.get(
                "/health",
                headers={"host": "analyzing-llm-rationale-staging-hy7gvnvt4a-uc.a.run.app"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_council_returns_structured_probability(self):
        with mock.patch.object(server_module, "_SCADS_MODEL_ALLOWLIST", {"test-model": {}}):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "model": "council",
                    "market_probability": 0.4,
                    "market_outcome": "Yes",
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_answer"], "Yes")
        self.assertEqual(payload["confidence"], 0.7)
        self.assertEqual(payload["market_analysis"]["model_probability"], 0.7)
        self.assertIn("[Council debate]", payload["rationale"])
        self.assertEqual(payload["model_key"], "council")
        self.assertEqual(len(self.provider.calls), 1)

    def test_predict_stream_council_uses_council_orchestration(self):
        with mock.patch.object(server_module, "_SCADS_MODEL_ALLOWLIST", {"test-model": {}}):
            response = self.client.post(
                "/predict/stream",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "model": "council",
                    "market_probability": 0.4,
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertIn('"model_key": "council"', response.text)
        self.assertIn("[Council debate]", response.text)
        self.assertNotIn("Streaming answer.", response.text)

    def test_council_provider_applies_member_timeout_to_scads_http(self):
        with (
            mock.patch.dict(os.environ, {"SCADS_AI_API_KEY": "test-key"}, clear=False),
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"test-model": "provider/test-model"},
            ),
            mock.patch.object(server_module, "_COUNCIL_MEMBER_TIMEOUT_S", 12.5),
        ):
            provider = server_module._council_provider("test-model")

        self.assertEqual(provider.model_name, "provider/test-model")
        self.assertEqual(provider.request_timeout_s, 12.5)
        provider._session.close()

    def test_predict_council_uses_healthy_member_when_peer_fails(self):
        healthy = FakeProvider()
        failing = FailingProvider()
        providers = {"healthy": healthy, "failing": failing}
        with (
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"healthy": {}, "failing": {}},
            ),
            mock.patch.object(
                server_module,
                "_council_provider",
                side_effect=lambda label: providers[label],
            ),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the healthy council member complete?",
                    "model": "council",
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["confidence"], 0.7)
        self.assertEqual(len(healthy.calls), 1)
        self.assertEqual(failing.calls, 1)

    def test_predict_council_runs_debate_when_two_members_succeed(self):
        first = FakeProvider()
        second = FakeProvider()
        providers = {"first": first, "second": second}
        with (
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"first": {}, "second": {}},
            ),
            mock.patch.object(
                server_module,
                "_council_provider",
                side_effect=lambda label: providers[label],
            ),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will two members complete both rounds?",
                    "model": "council",
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(first.calls), 2)
        self.assertEqual(len(second.calls), 2)

    def test_predict_council_returns_503_when_all_members_fail(self):
        failing = FailingProvider()
        with (
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"failing": {}},
            ),
            mock.patch.object(server_module, "_council_provider", return_value=failing),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will an unavailable council fail cleanly?",
                    "model": "council",
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["Retry-After"], "10")
        self.assertNotIn("upstream unavailable", response.json()["detail"])
        self.assertEqual(failing.calls, 1)

    def test_predict_council_enforces_configured_quorum(self):
        healthy = FakeProvider()
        failing = FailingProvider()
        providers = {"healthy": healthy, "failing": failing}
        with (
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"healthy": {}, "failing": {}},
            ),
            mock.patch.object(server_module, "_COUNCIL_MIN_SUCCESSFUL_MEMBERS", 2),
            mock.patch.object(
                server_module,
                "_council_provider",
                side_effect=lambda label: providers[label],
            ),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will one member satisfy a two-member quorum?",
                    "model": "council",
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 503)

    def test_predict_structured_503_names_the_failed_model(self):
        # Structured (chat_mode=False) requests intentionally skip the chat
        # fallback chain to keep the served model identity stable, so a
        # requested model's own outage must surface as-is -- but the caller
        # still needs to know *which* model failed, not just "unavailable".
        failing = FailingProvider("gpt-oss-120b")
        with mock.patch.dict(_state, {"provider": failing}):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "attach_evidence": False,
                    "chat_mode": False,
                },
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("gpt-oss-120b", response.json()["detail"])
        self.assertNotIn("upstream unavailable", response.json()["detail"])

    def test_predict_allows_anonymous_when_api_key_unset(self):
        with mock.patch.object(server_module, "_REQUIRED_API_KEY", None):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "market_probability": 0.4,
                    "market_outcome": "Yes",
                    "attach_evidence": False,
                    "chat_mode": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["market_analysis"]["model_probability"], 0.7)
        self._require_auth_mock.assert_not_called()

    def test_predict_stream_chat_returns_sse_chunks(self):
        response = self.client.post(
            "/predict/stream",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "chat_mode": True,
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        body = response.text
        self.assertIn("event: meta", body)
        self.assertIn('"prepare_ms":', body)
        self.assertIn("event: delta", body)
        self.assertIn('"first_delta_ms":', body)
        self.assertIn('"provider_first_delta_ms":', body)
        self.assertIn("Streaming ", body)
        self.assertIn("event: done", body)
        self.assertIn("Streaming answer.", body)
        self._require_auth_mock.assert_not_called()

    def test_predict_stream_chat_returns_probability_score(self):
        self.provider.stream_response = "Streaming forecast.\n[p:0.64]"
        response = self.client.post(
            "/predict/stream",
            json={
                "question": "Will the Fed cut rates before July 31, 2026?",
                "chat_mode": True,
                "attach_evidence": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"question_type": "chat"', response.text)
        self.assertIn('"model_probability": 0.64', response.text)
        self.assertIn('"rationale": "**Forecast: 64%**\\n\\nStreaming forecast."', response.text)

    def test_predict_stream_uses_response_cache_for_repeated_request(self):
        payload = {
            "question": "Will the Fed cut rates before July 31, 2026?",
            "chat_mode": True,
            "attach_evidence": False,
        }
        first = self.client.post("/predict/stream", json=payload)
        second = self.client.post("/predict/stream", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertIn('"cache_hit": true', second.text)
        self.assertIn('"provider_first_delta_ms": 0', second.text)
        self.assertIn("event: delta", second.text)
        self.assertIn("Streaming answer.", second.text)
        self.assertIn("event: done", second.text)

    def test_predict_stream_falls_back_when_provider_stream_is_empty(self):
        provider = EmptyStreamProvider()
        provider.response = {
            "type": "chat",
            "rationale": "Blocking fallback answer.",
        }
        _state["provider"] = provider

        with mock.patch.object(server_module, "_INTERACTIVE_DEFAULT_MODEL", ""):
            response = self.client.post(
                "/predict/stream",
                json={
                    "question": "Can you answer normally?",
                    "chat_mode": True,
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: delta", body)
        self.assertIn('"stream_empty_fallback": true', body)
        self.assertIn("Blocking fallback answer.", body)
        self.assertIn('"first_delta_ms":', body)
        self.assertIn('"provider_first_delta_ms":', body)
        self.assertIn("event: done", body)
        self.assertEqual(len(provider.calls), 2)

    def test_agent_analyze_stream_returns_sse_report(self):
        self.provider.stream_response = json.dumps(self.provider.response)
        response = self.client.post(
            "/agent/analyze/stream",
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "evidence_top_k": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"].split(";")[0], "text/event-stream")
        body = response.text
        self.assertIn("event: delta", body)
        self.assertIn("Evidence supports a yes forecast.", body)
        self.assertIn("event: done", body)
        self.assertIn('"report"', body)
        self.assertIn('"model_probability": 0.7', body)
        self.assertIn('"agent_run"', body)
        runs = self.client.get(
            "/agent/runs",
            headers={"Authorization": f"Bearer {_issue_session('test-user', 'test@example.com', 'Test User', '')}"},
        )
        self.assertEqual(runs.status_code, 200)
        self.assertEqual(runs.json()["runs"][0]["status"], "completed")

    def test_market_forecast_stream_returns_model_probability(self):
        token = _issue_session("stream-user", "user@example.com", "Stream User", "")
        self.provider.stream_response = json.dumps(self.provider.response)
        response = self.client.post(
            "/market/forecast/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "market_probability": 0.4,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        self.assertIn('"model_probability": 0.7', body)

    def test_predict_merges_supplied_venue_articles_with_fresh_news(self):
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
        self.assertEqual(
            payload["evidence_articles"][1]["title"],
            "Central bank signals policy shift",
        )
        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will event X happen?", 20)],
        )

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
                "chat_mode": False,
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
        self.assertIn("Current Time:", prompt)
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

    def test_simple_chat_skips_context_preparation_work(self):
        with (
            mock.patch.object(
                server_module,
                "_fetch_market_context",
                side_effect=AssertionError("simple chat should not fetch market context"),
            ),
            mock.patch.object(
                server_module,
                "_fetch_evidence_with_cache",
                side_effect=AssertionError("simple chat should not fetch evidence"),
            ),
            mock.patch.object(
                server_module,
                "_rag_search",
                side_effect=AssertionError("anonymous simple chat should not search KB"),
            ),
            mock.patch.object(
                server_module,
                "_read_live_track_record",
                side_effect=AssertionError("non-trading simple chat should not load track record"),
            ),
        ):
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
        self.assertEqual(len(self.provider.calls), 1)

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
            "venue_news_articles": [{
                "title": "Venue update on X",
                "source": "Polymarket",
            }],
        }
        with mock.patch.object(md, "fetch_polymarket", lambda slug=None, market_id=None: quote):
            response = self.client.get("/markets/polymarket?slug=will-x")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["platform"], "Polymarket")
        self.assertAlmostEqual(body["probability"], 0.62)
        self.assertEqual(len(body["outcomes"]), 2)
        self.assertEqual(body["venue_news_articles"][0]["title"], "Venue update on X")

    def test_markets_polymarket_requires_identifier(self):
        self.assertEqual(self.client.get("/markets/polymarket").status_code, 422)

    def test_predict_enriches_supplied_market_price_with_rules_and_market_news_query(self):
        import analyzing_llm_rationale.market_data as md

        quote = {
            "platform": "Polymarket",
            "ident": "will-x",
            "question": "Will X happen before December 31, 2026?",
            "market_url": "https://polymarket.com/market/will-x",
            "description": "Polymarket's current event background.",
            "resolution_criteria": "Resolves Yes only if the official source confirms X.",
            "outcome": "Yes",
            "probability": 0.61,
            "outcomes": [],
            "venue_news_articles": [{
                "title": "Official X status update",
                "source": "Polymarket",
                "summary": "The venue linked an official status update.",
            }],
        }
        with mock.patch.object(
            md,
            "fetch_polymarket",
            lambda slug=None, market_id=None: quote,
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Please analyze this market.",
                    "market_platform": "Polymarket",
                    "market_url": quote["market_url"],
                    "market_probability": 0.60,
                    "evidence_top_k": 2,
                    "chat_mode": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("Polymarket's current event background.", prompt)
        self.assertIn("Resolves Yes only if the official source confirms X.", prompt)
        self.assertIn("Official X status update", prompt)
        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will X happen before December 31, 2026?", 2)],
        )

    def test_predict_skips_market_enrichment_when_context_is_complete(self):
        import analyzing_llm_rationale.market_data as md

        with mock.patch.object(md, "fetch_polymarket") as fetch_polymarket:
            response = self.client.post(
                "/predict",
                json={
                    "question": "Please analyze this already-loaded market.",
                    "market_platform": "Polymarket",
                    "market_url": "https://polymarket.com/market/will-x",
                    "market_probability": 0.60,
                    "description": "Already supplied venue background.",
                    "resolution_criteria": "Already supplied venue rules.",
                    "attach_evidence": False,
                    "chat_mode": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        fetch_polymarket.assert_not_called()
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("Already supplied venue background.", prompt)
        self.assertIn("Already supplied venue rules.", prompt)

    def test_markets_kalshi_not_found_maps_to_404(self):
        import analyzing_llm_rationale.market_data as md

        def boom(ticker):
            raise md.MarketDataError("Kalshi market not found.")

        with mock.patch.object(md, "fetch_kalshi", boom):
            response = self.client.get("/markets/kalshi?ticker=NOPE")
        self.assertEqual(response.status_code, 404)

    def test_radar_endpoint_returns_markets(self):
        response = self.client.get("/radar")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("updated_at", payload)
        self.assertGreaterEqual(len(payload["markets"]), 1)
        self.assertIn("question", payload["markets"][0])

    def test_radar_endpoint_schedules_evidence_prefetch(self):
        live = {
            "generated_at": "2026-06-28T23:51:20+00:00",
            "edge_board": [{
                "platform": "Kalshi",
                "ident": "KXEXAMPLE",
                "question": "Will example happen?",
                "market_url": "https://kalshi.com/markets/example",
                "market_probability": 0.45,
                "model_probability": 0.62,
                "edge": 0.17,
                "abs_edge": 0.17,
            }],
        }
        scheduled = []

        def fake_spawn(awaitable):
            scheduled.append(awaitable)
            awaitable.close()

        with (
            mock.patch.object(server_module, "_read_edge_board_record", return_value=live),
            mock.patch.object(server_module, "_spawn_background", side_effect=fake_spawn),
        ):
            response = self.client.get("/radar?limit=1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(scheduled), 1)

    def test_radar_evidence_prefetch_warms_forecast_cache(self):
        market = server_module.RadarMarket(
            id="one",
            ident="KXEXAMPLE",
            platform="Kalshi",
            question="Will example happen?",
            market_probability=0.45,
            model_probability=0.62,
        )

        import asyncio
        asyncio.run(server_module._prefetch_radar_evidence([market]))

        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will example happen?", 3)],
        )
        cache_key = server_module._cache_key("evidence", "Will example happen?", 3)
        cached = _cache_get(cache_key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["title"], "Central bank signals policy shift")

    def test_radar_endpoint_includes_live_edge_board_metadata(self):
        live = {
            "generated_at": "2026-06-28T23:51:20+00:00",
            "model": "council",
            "n_snapshots_resolved": 184,
            "n_markets_resolved": 139,
            "n_markets_open": 30,
            "models_comparison": [{"model": "council", "n_snapshots_resolved": 184}],
            "paper_pnl": {"flat": {"roi": 0.12, "growth_curve": [100, 112]}},
            "lead_lag": {"n_markets": 12},
            "calibration": {"bins": []},
            "resolved_log": [{"question": "Resolved example?", "outcome": 1}],
            "edge_board": [{
                "platform": "Kalshi",
                "ident": "KXEXAMPLE",
                "question": "Will example happen?",
                "market_url": "https://kalshi.com/markets/example",
                "description": "Venue background.",
                "resolution_criteria": "Official rules.",
                "categories": ["Economics"],
                "market_probability": 0.4,
                "model_probability": 0.55,
                "edge": 0.15,
                "abs_edge": 0.15,
                "side": "YES",
                "domain": "macro",
                "horizon": "30d+",
            }],
        }
        with (
            mock.patch.object(server_module, "_read_live_track_record", return_value=live),
            mock.patch.object(server_module, "_read_mark_to_market_record", return_value=None),
            mock.patch.object(server_module, "_cache_get", return_value=None),
            mock.patch.object(server_module, "_cache_set"),
        ):
            response = self.client.get("/radar?limit=6")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generated_at"], live["generated_at"])
        self.assertEqual(payload["model"], "council")
        self.assertEqual(payload["n_snapshots_resolved"], 184)
        self.assertEqual(payload["n_markets_resolved"], 139)
        self.assertEqual(len(payload["edge_board"]), 1)
        self.assertEqual(payload["markets"][0]["ident"], "KXEXAMPLE")
        self.assertEqual(payload["markets"][0]["resolution_criteria"], "Official rules.")
        self.assertEqual(payload["models_comparison"][0]["model"], "council")
        self.assertEqual(payload["paper_pnl"]["flat"]["growth_curve"], [100, 112])
        self.assertEqual(payload["lead_lag"]["n_markets"], 12)
        self.assertEqual(payload["resolved_log"][0]["question"], "Resolved example?")
        self.assertEqual(payload["freshness"]["generated_at"], live["generated_at"])
        self.assertIn("no-cache", response.headers["cache-control"])
        self.assertEqual(payload["markets"][0]["question"], "Will example happen?")

    def test_edge_board_endpoint_includes_freshness_and_no_cache(self):
        live = {
            "generated_at": "2026-06-28T23:51:20+00:00",
            "edge_board": [{"question": "Live edge?", "edge": 0.2}],
            "paper_pnl": {"flat": {"growth_curve": [100, 105]}},
            "mark_to_market_account": {"account_value": 9999.47, "n_open_positions": 8},
            "mark_to_market_by_model": [{"model": "council", "account_value": 9999.47}],
            "mark_to_market_cycle_minutes": 15,
            "model": "council",
            "n_snapshots_resolved": 184,
            "n_markets_resolved": 139,
            "n_markets_open": 1,
            "resolved_log": [{"question": "Resolved edge?"}],
        }
        with (
            mock.patch.object(server_module, "_read_live_track_record", return_value=live),
            mock.patch.object(server_module, "_read_mark_to_market_record", return_value=None),
        ):
            response = self.client.get("/edge-board")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model"], "council")
        self.assertEqual(payload["edge_board"][0]["question"], "Live edge?")
        self.assertEqual(payload["paper_pnl"]["flat"]["growth_curve"], [100, 105])
        self.assertEqual(payload["mark_to_market_account"]["account_value"], 9999.47)
        self.assertEqual(payload["mark_to_market_by_model"][0]["model"], "council")
        self.assertEqual(payload["mark_to_market_cycle_minutes"], 15)
        self.assertEqual(payload["n_markets_resolved"], 139)
        self.assertEqual(payload["resolved_log"][0]["question"], "Resolved edge?")
        self.assertEqual(payload["freshness"]["generated_at"], live["generated_at"])
        self.assertIn("no-cache", response.headers["cache-control"])

    def test_agent_trading_board_endpoint_returns_leaderboard_and_freshness(self):
        live = {
            "generated_at": "2026-08-11T18:00:00+00:00",
            "models": ["gpt-oss-120b"],
            "leaderboard": [{"agent_id": "gpt-oss-120b", "account_value": 9981.46}],
            "equity_curves": {"gpt-oss-120b": {"value_curve": [{"account_value": 10000.0}]}},
            "recent_activity": [{"agent_id": "gpt-oss-120b", "type": "trade"}],
        }
        with mock.patch.object(server_module, "_read_agent_trading_board", return_value=live):
            response = self.client.get("/agent-trading/board")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "shadow")
        self.assertEqual(payload["models"], ["gpt-oss-120b"])
        self.assertEqual(payload["leaderboard"][0]["agent_id"], "gpt-oss-120b")
        self.assertEqual(payload["equity_curves"]["gpt-oss-120b"]["value_curve"][0]["account_value"], 10000.0)
        self.assertEqual(payload["recent_activity"][0]["type"], "trade")
        self.assertEqual(payload["freshness"]["generated_at"], live["generated_at"])
        self.assertIn("no-cache", response.headers["cache-control"])

    def test_agent_trading_board_endpoint_is_never_influenced_by_a_live_mode_field(self):
        # Shadow-only is a hard product guarantee for this feature -- the
        # endpoint must report "shadow" regardless of whatever the published
        # artifact itself might (incorrectly) claim, not pass through a
        # "mode" field from the live payload.
        live = {"generated_at": "now", "mode": "live", "leaderboard": []}
        with mock.patch.object(server_module, "_read_agent_trading_board", return_value=live):
            response = self.client.get("/agent-trading/board")
        self.assertEqual(response.json()["mode"], "shadow")

    def test_agent_trading_board_endpoint_defaults_to_empty_when_no_live_file(self):
        with mock.patch.object(server_module, "_read_agent_trading_board", return_value=None):
            response = self.client.get("/agent-trading/board")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["leaderboard"], [])
        self.assertEqual(payload["equity_curves"], {})
        self.assertEqual(payload["recent_activity"], [])
        self.assertEqual(payload["mode"], "shadow")

    def test_edge_board_endpoint_compacts_large_chart_payloads(self):
        long_curve = list(range(server_module._EDGE_BOARD_CURVE_MAX_POINTS + 40))
        live = {
            "generated_at": "2026-06-28T23:51:20+00:00",
            "paper_pnl": {
                "smart": {
                    "n_bets": 200,
                    "roi": 0.12,
                    "growth_curve": long_curve,
                    "equity_curve_ts": [f"2026-01-{(i % 28) + 1:02d}T00:00:00+00:00" for i in long_curve],
                    "bet_log": [{"private": True}] * 200,
                }
            },
            "models_comparison": [{
                "model": "council",
                "n_snapshots_resolved": 200,
                "accuracy": 0.7,
                "paper_pnl": {
                    "smart": {
                        "n_bets": 200,
                        "roi": 0.1,
                        "growth_curve": long_curve,
                        "bet_log": [{"private": True}] * 200,
                    }
                },
            }],
            "mark_to_market_account": {
                "account_value": 9999.47,
                "value_curve": [{"account_value": i, "ts": str(i)} for i in long_curve],
                "trades": [{"private": True}] * 200,
            },
            "mark_to_market_by_model": [{
                "model": "council",
                "account": {
                    "account_value": 9999.47,
                    "value_curve": [{"account_value": i, "ts": str(i)} for i in long_curve],
                    "open_positions": [{"private": True}] * 200,
                },
            }],
        }
        with (
            mock.patch.object(server_module, "_read_live_track_record", return_value=live),
            mock.patch.object(server_module, "_read_mark_to_market_record", return_value=None),
        ):
            response = self.client.get("/edge-board")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertLessEqual(
            len(payload["paper_pnl"]["smart"]["growth_curve"]),
            server_module._EDGE_BOARD_CURVE_MAX_POINTS,
        )
        self.assertEqual(payload["paper_pnl"]["smart"]["growth_curve"][0], long_curve[0])
        self.assertEqual(payload["paper_pnl"]["smart"]["growth_curve"][-1], long_curve[-1])
        self.assertNotIn("bet_log", payload["paper_pnl"]["smart"])
        self.assertNotIn("trades", payload["mark_to_market_account"])
        self.assertNotIn("open_positions", payload["mark_to_market_by_model"][0]["account"])
        self.assertLessEqual(
            len(payload["mark_to_market_by_model"][0]["account"]["value_curve"]),
            server_module._EDGE_BOARD_CURVE_MAX_POINTS,
        )
        self.assertNotIn("bet_log", payload["models_comparison"][0]["paper_pnl"]["smart"])

    def test_edge_board_endpoint_merges_independent_mtm_artifact(self):
        resolved = {
            "generated_at": "2026-06-28T23:00:00+00:00",
            "models_comparison": [{"model": "council"}],
            "paper_pnl": {"flat": {"growth_curve": [100, 101]}},
            "n_snapshots_resolved": 184,
            "n_markets_resolved": 139,
            "resolved_log": [{"question": "Resolved edge?"}],
        }
        mtm = {
            "generated_at": "2026-06-28T23:15:00+00:00",
            "edge_board": [{"question": "Fresh MTM edge?", "edge": 0.2}],
            "mark_to_market_account": {"account_value": 9999.47, "n_open_positions": 8},
            "mark_to_market_by_model": [{"model": "council", "account_value": 9999.47}],
            "mark_to_market_cycle_minutes": 15,
            "n_markets_open": 1,
            "n_markets_tracked": 8,
        }
        with (
            mock.patch.object(server_module, "_read_live_track_record", return_value=resolved),
            mock.patch.object(server_module, "_read_mark_to_market_record", return_value=mtm),
        ):
            response = self.client.get("/edge-board")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generated_at"], mtm["generated_at"])
        self.assertEqual(payload["freshness"]["generated_at"], mtm["generated_at"])
        self.assertEqual(payload["edge_board"][0]["question"], "Fresh MTM edge?")
        self.assertEqual(payload["mark_to_market_account"]["account_value"], 9999.47)
        self.assertEqual(payload["models_comparison"][0]["model"], "council")
        self.assertEqual(payload["resolved_log"][0]["question"], "Resolved edge?")

    def test_scads_allowlist_includes_board_models(self):
        from analyzing_llm_rationale.config import scads_hosted_model_allowlist

        expected = scads_hosted_model_allowlist(server_module._REPO_ROOT / "configs" / "models.yaml")
        self.assertEqual(server_module._SCADS_MODEL_ALLOWLIST, expected)
        self.assertIn("scads-alias-reasoning", server_module._SCADS_MODEL_ALLOWLIST)
        self.assertIn("kimi-k2.7-code", server_module._SCADS_MODEL_ALLOWLIST)
        self.assertIn("deepseek-v3", server_module._SCADS_MODEL_ALLOWLIST)
        self.assertIn("kimi-k2.6", server_module._SCADS_MODEL_ALLOWLIST)

    def test_analytics_event_summary_counts_events(self):
        response = self.client.post(
            "/analytics/event",
            json={"event_name": "forecast_completed", "path": "/", "metadata": {"source": "test"}},
        )
        self.assertEqual(response.status_code, 200)
        time.sleep(0.02)

        summary = self.client.get("/analytics/events/summary")
        self.assertEqual(summary.status_code, 200)
        payload = summary.json()
        self.assertEqual(payload["total_events"], 1)
        self.assertEqual(payload["by_event"][0]["event_name"], "forecast_completed")

    def test_share_forecast_creates_public_page(self):
        response = self.client.post(
            "/forecasts/share",
            json={
                "question": "Will the Fed cut rates before September 30, 2026?",
                "predicted_answer": "Yes",
                "confidence": 0.7,
                "rationale": "Evidence supports a yes forecast.",
                "model_probability": 0.7,
                "market_probability": 0.42,
                "market_platform": "Polymarket",
            },
        )
        self.assertEqual(response.status_code, 200)
        share = response.json()
        self.assertIn("/forecast/", share["url"])

        page = self.client.get(f"/forecast/{share['share_id']}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Will the Fed cut rates", page.text)
        self.assertIn("Evidence supports", page.text)

    def test_fetch_evidence_auto_initializes_pipeline(self):
        import asyncio

        from analyzing_llm_rationale.server import _fetch_evidence_with_cache, _local_cache, _state
        _state.pop("evidence_pipeline", None)
        _local_cache.clear()
        # Should auto-initialize rather than returning 'unconfigured' error
        with unittest.mock.patch("analyzing_llm_rationale.news_pipeline.NewsPipeline.__init__", return_value=None), \
             unittest.mock.patch("analyzing_llm_rationale.news_pipeline.NewsPipeline.fetch_summarize_rank", return_value=[]):
            articles, error, outcome = asyncio.run(_fetch_evidence_with_cache("Will X happen auto init?", 3, source="test"))
            self.assertIsNotNone(_state.get("evidence_pipeline"))
            self.assertNotIn("unconfigured", (error or "").lower())

    def test_source_attribution_appended_for_typed_predict(self):
        import asyncio

        from analyzing_llm_rationale.server import (
            EvidenceSource,
            PredictRequest,
            PredictResponse,
            _finalize_predict_response,
        )
        req = PredictRequest(question="Will X happen?", attach_evidence=True, chat_mode=False)
        resp = PredictResponse(
            question_type="binary",
            predicted_answer="Yes",
            confidence=0.7,
            variant="variant0_neutral_baseline",
            model_key="gpt-oss-120b",
            rationale="Initial rationale text.",
            evidence_sources=[
                EvidenceSource(source="Reuters", title="Event X update", relevance_score=0.9)
            ]
        )
        asyncio.run(_finalize_predict_response(req, resp, None))
        self.assertIn("**Sources provided to the forecast**", resp.rationale)
        self.assertIn("Reuters", resp.rationale)

    def test_trading_accounts_requires_session(self):
        response = self.client.get("/trading/accounts")
        self.assertEqual(response.status_code, 401)

    def test_trading_preview_requires_session(self):
        response = self.client.post(
            "/trading/preview",
            json={"platform": "kalshi", "ticker": "KXTEST", "price": 0.5, "quantity": 1},
        )
        self.assertEqual(response.status_code, 401)

    def test_authenticated_trading_preview_returns_normalized_order(self):
        token = _issue_session("trader-1", "trader@example.com", "Trader", "")
        response = self.client.post(
            "/trading/preview",
            json={
                "platform": "kalshi",
                "ticker": "KXTEST",
                "action": "buy",
                "outcome": "yes",
                "price": 0.44,
                "quantity": 2,
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["would_execute"])
        self.assertEqual(body["normalized_order"]["exchange_order"]["side"], "bid")

    def test_trading_order_disabled_by_default(self):
        token = _issue_session("trader-1", "trader@example.com", "Trader", "")
        with mock.patch.dict(os.environ, {"FORESEA_ENABLE_TRADING": "false"}, clear=False):
            response = self.client.post(
                "/trading/orders",
                json={
                    "platform": "kalshi",
                    "ticker": "KXTEST",
                    "price": 0.50,
                    "quantity": 1,
                    "execute": True,
                    "confirmation": "PLACE REAL ORDER",
                },
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("Live trading is disabled", response.json()["detail"])

    def test_trading_accounts_check_reports_byo_request_source(self):
        token = _issue_session("trader-1", "trader@example.com", "Trader", "")
        response = self.client.post(
            "/trading/accounts/check",
            json={"venue_credentials": {
                "kalshi_api_key_id": "byo-key",
                "kalshi_private_key": "byo-secret-pem",
            }},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["credential_source"], "request")
        self.assertTrue(body["venues"]["kalshi"]["configured"])
        # Supplied secrets are never echoed back.
        self.assertNotIn("byo-secret-pem", response.text)

    def test_trading_preview_does_not_echo_supplied_credentials(self):
        token = _issue_session("trader-1", "trader@example.com", "Trader", "")
        response = self.client.post(
            "/trading/preview",
            json={
                "platform": "kalshi",
                "ticker": "KXTEST",
                "action": "buy",
                "outcome": "yes",
                "price": 0.44,
                "quantity": 2,
                "venue_credentials": {
                    "kalshi_api_key_id": "byo-key",
                    "kalshi_private_key": "byo-secret-pem",
                },
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("byo-secret-pem", response.text)
        self.assertIn("Connect the account securely first", response.json()["detail"])

    def test_secure_trading_connection_encrypts_credentials_and_never_returns_them(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")
        token = _issue_session("secure-trader", "secure@example.com", "Secure", "")
        headers = {"Authorization": f"Bearer {token}"}
        kms = FakeTradingKms()
        kms_key_name = "projects/test-project/locations/us-central1/keyRings/foresea/cryptoKeys/trading"
        with mock.patch.dict(
            os.environ,
            {"FORESEA_TRADING_KMS_KEY_NAME": kms_key_name},
            clear=False,
        ), mock.patch.object(server_module, "_get_trading_kms_client", return_value=kms):
            created = self.client.put(
                "/trading/connections/kalshi",
                json={"venue_credentials": {
                    "kalshi_api_key_id": "secure-key-id",
                    "kalshi_private_key": pem,
                }},
                headers=headers,
            )
            self.assertEqual(created.status_code, 200)
            self.assertEqual(created.json()["platform"], "kalshi")
            self.assertTrue(created.json()["connected"])
            self.assertNotIn(pem, created.text)
            self.assertNotIn("secure-key-id", created.text)
            stored = _state["trading_connections"]["secure-trader"]["kalshi"]
            self.assertEqual(stored["credential_version"], 2)
            self.assertEqual(stored["kms_key_name"], kms_key_name)
            self.assertNotIn(pem, stored["encrypted_credentials"])
            self.assertNotIn("secure-key-id", stored["encrypted_credentials"])
            self.assertNotIn(pem, stored["wrapped_data_key"])
            self.assertNotIn("secure-key-id", stored["wrapped_data_key"])
            self.assertEqual(len(kms.encrypt_requests), 1)
            self.assertEqual(
                kms.encrypt_requests[0]["additional_authenticated_data"],
                b"foresea:trading-connection:v2:secure-trader:kalshi",
            )

            listed = self.client.get("/trading/connections", headers=headers)
            self.assertEqual(listed.status_code, 200)
            self.assertTrue(listed.json()["connections"]["kalshi"]["connected"])
            self.assertNotIn(pem, listed.text)

            with mock.patch(
                "analyzing_llm_rationale.trading.preview_order",
                wraps=trading_module.preview_order,
            ) as preview_order:
                preview = self.client.post(
                    "/trading/preview",
                    json={"platform": "kalshi", "ticker": "KXTEST", "price": 0.5, "quantity": 1},
                    headers=headers,
                )
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview_order.call_args.args[1]["kalshi_api_key_id"], "secure-key-id")
            self.assertEqual(len(kms.decrypt_requests), 1)

            removed = self.client.delete("/trading/connections/kalshi", headers=headers)
            self.assertEqual(removed.status_code, 200)
            self.assertFalse(removed.json()["connected"])

    def test_legacy_trading_connection_is_rewrapped_with_a_kms_data_key(self):
        from cryptography.fernet import Fernet

        legacy_key = Fernet.generate_key()
        legacy_credentials = {"kalshi_api_key_id": "legacy-key", "kalshi_private_key": "legacy-pem"}
        legacy_ciphertext = Fernet(legacy_key).encrypt(
            json.dumps(legacy_credentials, sort_keys=True).encode("utf-8")
        ).decode("utf-8")
        _state.setdefault("trading_connections", {}).setdefault("legacy-trader", {})["kalshi"] = {
            "platform": "kalshi",
            "encrypted_credentials": legacy_ciphertext,
            "credential_version": 1,
        }
        kms = FakeTradingKms()
        with mock.patch.dict(
            os.environ,
            {
                "FORESEA_CREDENTIALS_ENCRYPTION_KEY": legacy_key.decode("utf-8"),
                "FORESEA_TRADING_KMS_KEY_NAME": (
                    "projects/test-project/locations/us-central1/keyRings/foresea/cryptoKeys/trading"
                ),
            },
            clear=False,
        ), mock.patch.object(server_module, "_get_trading_kms_client", return_value=kms):
            self.assertEqual(
                server_module._stored_trading_credentials("legacy-trader", "kalshi"),
                legacy_credentials,
            )
        migrated = _state["trading_connections"]["legacy-trader"]["kalshi"]
        self.assertEqual(migrated["credential_version"], 2)
        self.assertIn("wrapped_data_key", migrated)
        self.assertNotEqual(migrated["encrypted_credentials"], legacy_ciphertext)
        self.assertNotIn("legacy-pem", migrated["encrypted_credentials"])

    def test_submitted_order_is_audited_and_can_be_reconciled_or_cancelled(self):
        token = _issue_session("audit-trader", "audit@example.com", "Audit", "")
        headers = {"Authorization": f"Bearer {token}"}
        fake_connection = {
            "kalshi_api_key_id": "safe-test-key",
            "kalshi_private_key": "safe-test-pem",
            "kalshi_base_url": "https://external-api.kalshi.com/trade-api/v2",
        }
        fake_result = {
            "ok": True,
            "platform": "kalshi",
            "would_execute": True,
            "requires_confirmation": True,
            "confirmation_phrase": "PLACE REAL ORDER",
            "trading_enabled": True,
            "max_order_notional": 50.0,
            "estimated_notional": 1.0,
            "warnings": [],
            "normalized_order": {
                "platform": "kalshi", "action": "buy", "outcome": "yes",
                "order_type": "limit", "ticker": "KXTEST", "quantity": 2.0,
                "price": 0.5, "subaccount": 0, "exchange_index": 0,
            },
            "submitted": True,
            "user_id": "audit-trader",
            "venue_response": {"body": {"order_id": "venue-123", "status": "resting"}},
        }
        with mock.patch.object(server_module, "_stored_trading_credentials", return_value=fake_connection), mock.patch(
            "analyzing_llm_rationale.trading.place_order", return_value=fake_result
        ), mock.patch.object(
            server_module, "_validate_live_trade_guardrails", new=mock.AsyncMock(return_value={})
        ):
            submitted = self.client.post(
                "/trading/orders",
                json={
                    "platform": "kalshi", "ticker": "KXTEST", "price": 0.5,
                    "quantity": 2, "execute": True, "confirmation": "PLACE REAL ORDER",
                },
                headers=headers,
            )
            self.assertEqual(submitted.status_code, 200)
            audit_id = submitted.json()["audit_order_id"]
            self.assertEqual(submitted.json()["reconciliation_status"], "open")

            listed = self.client.get("/trading/orders", headers=headers)
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["orders"][0]["venue_order_id"], "venue-123")

            with mock.patch(
                "analyzing_llm_rationale.trading.reconcile_order",
                return_value={
                    "venue_order_id": "venue-123", "status": "open", "venue_status": "resting",
                    "filled_quantity": 1.0, "remaining_quantity": 1.0,
                },
            ):
                reconciled = self.client.post(f"/trading/orders/{audit_id}/reconcile", headers=headers)
            self.assertEqual(reconciled.status_code, 200)
            self.assertEqual(reconciled.json()["filled_quantity"], 1.0)

            with mock.patch(
                "analyzing_llm_rationale.trading.cancel_order",
                return_value={
                    "venue_order_id": "venue-123", "status": "canceled", "venue_status": "canceled",
                    "remaining_quantity": 0.0,
                },
            ):
                cancelled = self.client.request(
                    "DELETE",
                    f"/trading/orders/{audit_id}",
                    json={"confirmation": "CANCEL OPEN ORDER"},
                    headers=headers,
                )
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["status"], "canceled")

    def test_trade_run_requires_explicit_execution_and_tracks_reconciliation(self):
        token = _issue_session("run-trader", "run@example.com", "Run", "")
        headers = {"Authorization": f"Bearer {token}"}
        fake_connection = {
            "kalshi_api_key_id": "safe-test-key",
            "kalshi_private_key": "safe-test-pem",
            "kalshi_base_url": "https://external-api.kalshi.com/trade-api/v2",
        }
        fake_result = {
            "ok": True,
            "platform": "kalshi",
            "would_execute": True,
            "requires_confirmation": True,
            "confirmation_phrase": "PLACE REAL ORDER",
            "trading_enabled": True,
            "max_order_notional": 50.0,
            "estimated_notional": 1.0,
            "warnings": [],
            "normalized_order": {
                "platform": "kalshi", "action": "buy", "outcome": "yes",
                "order_type": "limit", "ticker": "KXTEST", "quantity": 2.0,
                "price": 0.5, "subaccount": 0, "exchange_index": 0,
            },
            "submitted": True,
            "user_id": "run-trader",
            "venue_response": {"body": {"order_id": "run-venue-123", "status": "resting"}},
        }
        with mock.patch.object(
            server_module, "_stored_trading_credentials", return_value=fake_connection
        ):
            created = self.client.post(
                "/trading/runs",
                json={
                    "platform": "kalshi", "ticker": "KXTEST", "price": 0.5,
                    "quantity": 2, "title": "Fed easing thesis", "thesis": "Policy data favors YES.",
                    "expected_edge": 0.08, "sources": ["https://example.test/source"],
                },
                headers=headers,
            )
            self.assertEqual(created.status_code, 200)
            run = created.json()
            run_id = run["id"]
            self.assertEqual(run["status"], "awaiting_approval")
            self.assertTrue(run["client_order_id"].startswith("foresea-run-"))
            self.assertEqual(run["provenance"]["expected_edge"], 0.08)
            self.assertNotIn("safe-test-pem", created.text)

            listed = self.client.get("/trading/runs", headers=headers)
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["runs"][0]["id"], run_id)

            rejected = self.client.post(
                f"/trading/runs/{run_id}/execute",
                json={"confirmation": "not approved"},
                headers=headers,
            )
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(
                self.client.get(f"/trading/runs/{run_id}", headers=headers).json()["status"],
                "awaiting_approval",
            )

            with mock.patch(
                "analyzing_llm_rationale.trading.place_order", return_value=fake_result
            ), mock.patch.object(
                server_module, "_validate_live_trade_guardrails", new=mock.AsyncMock(return_value={})
            ):
                submitted = self.client.post(
                    f"/trading/runs/{run_id}/execute",
                    json={"confirmation": "PLACE REAL ORDER"},
                    headers=headers,
                )
            self.assertEqual(submitted.status_code, 200)
            self.assertEqual(submitted.json()["status"], "submitted")
            self.assertEqual(submitted.json()["venue_order_id"], "run-venue-123")
            audit_id = submitted.json()["audit_order_id"]
            self.assertEqual(
                self.client.get("/trading/orders", headers=headers).json()["orders"][0]["trade_run_id"],
                run_id,
            )

            with mock.patch(
                "analyzing_llm_rationale.trading.reconcile_order",
                return_value={
                    "venue_order_id": "run-venue-123", "status": "filled", "venue_status": "filled",
                    "filled_quantity": 2.0, "remaining_quantity": 0.0,
                },
            ):
                reconciled = self.client.post(
                    f"/trading/runs/{run_id}/reconcile", headers=headers
                )
            self.assertEqual(reconciled.status_code, 200)
            self.assertEqual(reconciled.json()["status"], "filled")
            self.assertEqual(reconciled.json()["audit_order_id"], audit_id)

            duplicate = self.client.post(
                f"/trading/runs/{run_id}/execute",
                json={"confirmation": "PLACE REAL ORDER"},
                headers=headers,
            )
            self.assertEqual(duplicate.status_code, 409)

    def test_trade_run_execution_claim_has_one_in_memory_owner(self):
        req = server_module.TradeRunCreateRequest(
            platform="kalshi", ticker="KXTEST", price=0.5, quantity=2
        )
        preview = {
            "estimated_notional": 1.0,
            "normalized_order": {
                "platform": "kalshi", "action": "buy", "outcome": "yes",
                "order_type": "limit", "ticker": "KXTEST", "quantity": 2.0,
                "price": 0.5,
            },
        }
        run = server_module._put_trading_run(
            "claim-user",
            server_module._new_trading_run(
                req,
                {"platform": "kalshi", "ticker": "KXTEST", "price": 0.5, "quantity": 2},
                preview,
            ),
        )

        first, first_owner = server_module._claim_trading_run_for_execution(
            "claim-user", run["id"], preview
        )
        second, second_owner = server_module._claim_trading_run_for_execution(
            "claim-user", run["id"], preview
        )

        self.assertTrue(first_owner)
        self.assertFalse(second_owner)
        self.assertEqual(first["status"], "submitting")
        self.assertEqual(second["status"], "submitting")

    def test_trading_guardrails_are_user_scoped_and_can_only_be_narrowed(self):
        token = _issue_session("risk-user", "risk@example.com", "Risk", "")
        headers = {"Authorization": f"Bearer {token}"}

        initial = self.client.get("/trading/guardrails", headers=headers)
        self.assertEqual(initial.status_code, 200)
        self.assertFalse(initial.json()["paused"])
        self.assertFalse(initial.json()["platform_kill_switch"])

        updated = self.client.put(
            "/trading/guardrails",
            json={
                "paused": True,
                "max_order_notional": 0.75,
                "max_daily_risk_notional": 2.0,
                "max_market_exposure_notional": 1.5,
                "max_price_deviation_bps": 100,
            },
            headers=headers,
        )
        self.assertEqual(updated.status_code, 200)
        self.assertTrue(updated.json()["paused"])
        self.assertEqual(updated.json()["max_order_notional"], 0.75)
        self.assertEqual(updated.json()["max_price_deviation_bps"], 100)

        above_cap = self.client.put(
            "/trading/guardrails",
            json={"max_order_notional": 100_000},
            headers=headers,
        )
        self.assertEqual(above_cap.status_code, 409)
        self.assertIn("hard ceiling", above_cap.json()["detail"])

        events = self.client.get("/trading/guardrails/events", headers=headers)
        self.assertEqual(events.status_code, 200)
        self.assertEqual(events.json()["events"][0]["event"], "guardrail_updated")

    def test_live_guardrails_block_price_deviation_and_daily_risk(self):
        user_id = "guardrail-user"
        payload = {
            "platform": "kalshi", "ticker": "KXTEST", "action": "buy", "outcome": "yes",
            "price": 0.60, "quantity": 2,
        }
        preview = {
            "estimated_notional": 1.2,
            "normalized_order": {
                "platform": "kalshi", "ticker": "KXTEST", "action": "buy", "outcome": "yes",
                "price": 0.60, "quantity": 2.0,
            },
        }
        self.assertEqual(
            server_module._update_trading_guardrails(
                user_id, server_module.TradingGuardrailsUpdateRequest(max_price_deviation_bps=100)
            )["max_price_deviation_bps"],
            100,
        )
        with mock.patch.object(
            server_module,
            "_fresh_trade_guard_quote",
            new=mock.AsyncMock(return_value={
                "outcome_probability": 0.50,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "market_ident": "KXTEST",
            }),
        ):
            with self.assertRaises(server_module.TradingGuardrailError) as price_error:
                asyncio.run(
                    server_module._validate_live_trade_guardrails(
                        user_id, payload=payload, preview=preview, credentials={"kalshi_api_key_id": "safe"}
                    )
                )
        self.assertEqual(price_error.exception.code, "price_deviation")

        server_module._update_trading_guardrails(
            user_id,
            server_module.TradingGuardrailsUpdateRequest(
                max_price_deviation_bps=300,
                max_daily_risk_notional=1.5,
            ),
        )
        server_module._put_trading_order(
            user_id,
            {
                "id": "daily-risk-order", "platform": "kalshi", "ticker": "KXOLD", "status": "filled",
                "action": "buy", "outcome": "yes", "quantity": 2.0, "price": 0.5,
                "estimated_notional": 1.0, "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        with mock.patch.object(
            server_module,
            "_fresh_trade_guard_quote",
            new=mock.AsyncMock(return_value={
                "outcome_probability": 0.60,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "market_ident": "KXTEST",
            }),
        ), mock.patch(
            "analyzing_llm_rationale.trading.reconcile_portfolio",
            return_value={"balance": {"available": 100.0, "unit": "USDC"}, "positions": []},
        ):
            with self.assertRaises(server_module.TradingGuardrailError) as daily_error:
                asyncio.run(
                    server_module._validate_live_trade_guardrails(
                        user_id, payload=payload, preview=preview, credentials={"kalshi_api_key_id": "safe"}
                    )
                )
        self.assertEqual(daily_error.exception.code, "daily_risk_limit")

    def test_trade_run_kill_switch_blocks_before_exchange_submission(self):
        token = _issue_session("kill-switch-user", "kill@example.com", "Kill", "")
        headers = {"Authorization": f"Bearer {token}"}
        connection = {"kalshi_api_key_id": "safe-key", "kalshi_private_key": "safe-pem"}
        with mock.patch.dict(
            os.environ,
            {"FORESEA_ENABLE_BYO_TRADING": "true", "FORESEA_TRADING_KILL_SWITCH": "true"},
            clear=False,
        ), mock.patch.object(server_module, "_stored_trading_credentials", return_value=connection), mock.patch(
            "analyzing_llm_rationale.trading.place_order"
        ) as place_order:
            created = self.client.post(
                "/trading/runs",
                json={"platform": "kalshi", "ticker": "KXTEST", "price": 0.5, "quantity": 1},
                headers=headers,
            )
            self.assertEqual(created.status_code, 200)
            blocked = self.client.post(
                f"/trading/runs/{created.json()['id']}/execute",
                json={"confirmation": "PLACE REAL ORDER"},
                headers=headers,
            )
        self.assertEqual(blocked.status_code, 409)
        self.assertIn("kill switch", blocked.json()["detail"])
        place_order.assert_not_called()
        events = self.client.get("/trading/guardrails/events", headers=headers).json()["events"]
        self.assertTrue(any(event["reason_code"] == "platform_kill_switch" for event in events))

    def test_scheduled_reconciliation_is_token_gated_and_updates_open_orders(self):
        order = server_module._put_trading_order(
            "scheduled-user",
            {
                "id": "scheduled-order",
                "trade_run_id": None,
                "platform": "kalshi",
                "venue_order_id": "venue-scheduled",
                "status": "open",
                "venue_status": "resting",
                "action": "buy",
                "outcome": "yes",
                "ticker": "KXTEST",
                "quantity": 2.0,
                "price": 0.5,
                "created_at": "2026-08-15T00:00:00+00:00",
                "updated_at": "2026-08-15T00:00:00+00:00",
            },
        )
        self.assertEqual(order["status"], "open")
        with mock.patch.object(server_module, "_TRADING_RECONCILIATION_TOKEN", "scheduler-token"):
            denied = self.client.post("/internal/trading/reconcile")
            self.assertEqual(denied.status_code, 401)
            with mock.patch.object(
                server_module, "_stored_trading_credentials", return_value={"kalshi_api_key_id": "key"}
            ), mock.patch(
                "analyzing_llm_rationale.trading.reconcile_order",
                return_value={
                    "venue_order_id": "venue-scheduled", "status": "filled", "venue_status": "filled",
                    "filled_quantity": 2.0, "remaining_quantity": 0.0,
                },
            ):
                reconciled = self.client.post(
                    "/internal/trading/reconcile?limit=10",
                    headers={"X-Trading-Reconciliation-Token": "scheduler-token"},
                )
        self.assertEqual(reconciled.status_code, 200)
        self.assertEqual(reconciled.json()["checked"], 1)
        self.assertEqual(reconciled.json()["updated"], 1)
        self.assertEqual(reconciled.json()["terminal"], 1)
        stored = server_module._read_trading_order("scheduled-user", "scheduled-order")
        self.assertEqual(stored["status"], "filled")
        self.assertEqual(stored["filled_quantity"], 2.0)

    def test_trading_launch_readiness_is_token_gated_and_does_not_expose_secrets(self):
        kms_key = (
            "projects/foresea/locations/us-central1/keyRings/foresea-trading/"
            "cryptoKeys/exchange-connections"
        )
        with mock.patch.object(server_module, "_TRADING_RECONCILIATION_TOKEN", "readiness-token"):
            denied = self.client.get("/internal/trading/readiness")
            self.assertEqual(denied.status_code, 401)
            with mock.patch.object(server_module, "_get_datastore", return_value=object()), mock.patch.dict(
                os.environ,
                {
                    "FORESEA_TRADING_KMS_KEY_NAME": kms_key,
                    "FORESEA_ENABLE_BYO_TRADING": "false",
                    "FORESEA_ENABLE_TRADING": "false",
                    "FORESEA_TRADING_KILL_SWITCH": "false",
                    "FORESEA_ALLOW_MARKET_ORDERS": "false",
                },
                clear=False,
            ):
                ready = self.client.get(
                    "/internal/trading/readiness",
                    headers={"X-Trading-Reconciliation-Token": "readiness-token"},
                )

        self.assertEqual(ready.status_code, 200)
        report = ready.json()
        self.assertTrue(report["safe_default_active"])
        self.assertTrue(report["ready_for_connection_beta"])
        self.assertFalse(report["ready_for_live_byo_beta"])
        self.assertTrue(report["scheduled_reconciliation_configured"])
        self.assertNotIn("readiness-token", ready.text)
        self.assertNotIn(kms_key, ready.text)
        self.assertEqual(
            {check["status"] for check in report["checks"] if check["code"] == "secure_connections"},
            {"ready"},
        )

    def test_copied_agent_profile_is_private_versioned_and_research_only(self):
        token = _issue_session("profile-user", "profile@example.com", "Profile", "")
        headers = {"Authorization": f"Bearer {token}"}
        with mock.patch.object(server_module, "_SCADS_MODEL_ALLOWLIST", {"test-model": "test-model"}):
            denied = self.client.post(
                "/agent-profiles/copy", json={"source_agent_id": "not-public"}, headers=headers
            )
            self.assertEqual(denied.status_code, 422)

            copied = self.client.post(
                "/agent-profiles/copy",
                json={"source_agent_id": "test-model", "name": "My copied model"},
                headers=headers,
            )
            self.assertEqual(copied.status_code, 201)
            payload = copied.json()
            profile = payload["profile"]
            self.assertTrue(payload["created"])
            self.assertEqual(profile["version"], 1)
            self.assertEqual(profile["execution_mode"], "research_only")
            self.assertNotIn("connection", json.dumps(profile).lower())

            copied_again = self.client.post(
                "/agent-profiles/copy",
                json={"source_agent_id": "test-model"},
                headers=headers,
            )
            self.assertEqual(copied_again.status_code, 201)
            self.assertFalse(copied_again.json()["created"])
            self.assertEqual(copied_again.json()["profile"]["id"], profile["id"])

            report = self.client.post(
                "/agent/analyze",
                json={
                    "question": "Will it rain tomorrow?",
                    "agent_profile_id": profile["id"],
                    "model": "attacker-controlled-model",
                    "tool_loop": True,
                    "benchmark_tools": True,
                },
                headers=headers,
            )

        self.assertEqual(report.status_code, 200)
        report_payload = report.json()
        self.assertIn("agent_profile", report_payload["pipeline"])
        self.assertNotIn("tool_loop", report_payload["pipeline"])
        self.assertEqual(report_payload["agent_profile"]["id"], profile["id"])
        self.assertEqual(report_payload["agent_profile"]["execution_mode"], "research_only")
        listed = self.client.get("/agent-profiles", headers=headers)
        self.assertEqual([item["id"] for item in listed.json()["profiles"]], [profile["id"]])
        deleted = self.client.delete(f"/agent-profiles/{profile['id']}", headers=headers)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/agent-profiles", headers=headers).json()["profiles"], [])

    def test_agent_run_is_private_durable_and_excludes_provider_secrets(self):
        owner_headers = {"Authorization": f"Bearer {_issue_session('run-owner', 'owner@example.com', 'Owner', '')}"}
        other_headers = {"Authorization": f"Bearer {_issue_session('run-other', 'other@example.com', 'Other', '')}"}
        report_response = self.client.post(
            "/agent/analyze",
            json={
                "question": "Will it rain tomorrow?",
                "openrouter_api_key": "never-persist-this-provider-secret",
                "history": [{"role": "user", "content": "private prior chat"}],
                "builtin_skills": True,
            },
            headers=owner_headers,
        )

        self.assertEqual(report_response.status_code, 200)
        report = report_response.json()
        run_id = report["agent_run"]["id"]
        self.assertEqual(report["agent_run"]["status"], "completed")

        listed = self.client.get("/agent/runs", headers=owner_headers)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["runs"][0]["id"], run_id)
        self.assertEqual(listed.json()["runs"][0]["status"], "completed")
        self.assertEqual(
            [event["phase"] for event in listed.json()["runs"][0]["timeline"]],
            ["created", "context_ready", "completed"],
        )

        detail = self.client.get(f"/agent/runs/{run_id}", headers=owner_headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["report"]["agent_run"]["id"], run_id)
        self.assertNotIn("never-persist-this-provider-secret", detail.text)
        self.assertNotIn("private prior chat", detail.text)
        self.assertNotIn("openrouter_api_key", detail.json()["request"])
        self.assertNotIn("history", detail.json()["request"])

        self.assertEqual(self.client.get(f"/agent/runs/{run_id}", headers=other_headers).status_code, 404)

    def test_favorites_crud_roundtrip(self):
        token = _issue_session("fav-user", "fav@example.com", "Fav", "")
        headers = {"Authorization": f"Bearer {token}"}
        key = "kalshi:KXTEST"
        # Empty to start.
        r0 = self.client.get("/favorites", headers=headers)
        self.assertEqual(r0.status_code, 200)
        self.assertEqual(r0.json()["favorites"], [])
        # Add one.
        r1 = self.client.put(
            f"/favorites/{key}",
            json={"key": key, "question": "Will X happen?", "platform": "kalshi",
                  "ident": "KXTEST", "notify": True},
            headers=headers,
        )
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.json()["createdAt"])
        # List shows it.
        r2 = self.client.get("/favorites", headers=headers)
        self.assertEqual(len(r2.json()["favorites"]), 1)
        self.assertEqual(r2.json()["favorites"][0]["key"], key)
        # Delete it.
        r3 = self.client.delete(f"/favorites/{key}", headers=headers)
        self.assertEqual(r3.status_code, 200)
        self.assertEqual(self.client.get("/favorites", headers=headers).json()["favorites"], [])

    def test_favorites_require_auth(self):
        self.assertEqual(self.client.get("/favorites").status_code, 401)

    def test_markets_search_merges_venues(self):
        from analyzing_llm_rationale import market_data as md
        poly = [{"platform": "Polymarket", "ident": "slug-a", "question": "Will A?",
                 "market_url": "u", "probability": 0.4, "close_time": None, "volume": 9}]
        kal = [{"platform": "Kalshi", "ident": "KXB", "question": "Will B?",
                "market_url": "u2", "probability": 0.6, "close_time": None, "volume": 3}]
        with mock.patch.object(md, "list_polymarket", return_value=poly), \
             mock.patch.object(md, "list_kalshi", return_value=kal):
            r = self.client.get("/markets/search", params={"q": "will", "limit": 10})
        self.assertEqual(r.status_code, 200)
        qs = {x["question"] for x in r.json()["results"]}
        self.assertEqual(qs, {"Will A?", "Will B?"})

    def test_watchlist_route_serves_spa(self):
        r = self.client.get("/watchlist")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("content-type", ""))

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

    def test_agent_analyze_benchmark_tool_loop_exposes_only_benchmark_tools(self):
        self.provider.response = {
            "thought": "remember this",
            "action": "manage_notes",
            "args": {"action": "add", "text": "Track Fed dates."},
        }
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
            os.environ,
            {"FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json")},
            clear=False,
        ):
            response = self.client.post(
                "/agent/analyze",
                json={
                    "question": "Will the Fed cut rates tomorrow?",
                    "tool_loop": True,
                    "benchmark_tools": True,
                    "max_tool_steps": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        report = response.json()
        self.assertEqual(report["pipeline"], ["tool_loop", "benchmark_tools"])
        self.assertEqual(report["tool_transcript"][0]["action"], "manage_notes")
        system_prompt = self.provider.calls[0][0]["content"]
        self.assertIn("place_trade(", system_prompt)
        self.assertIn("web_search(", system_prompt)
        self.assertIn("manage_notes(", system_prompt)
        self.assertNotIn("forecast(", system_prompt)
        self.assertNotIn("scan_markets(", system_prompt)

    def test_agent_analyze_tool_loop_without_benchmark_tools_excludes_place_trade(self):
        # Regression test: _agent_tool_loop used to merge benchmark_tool_map
        # (place_trade, web_search, manage_notes) into the standard tool set
        # unconditionally, regardless of req.benchmark_tools -- contradicting
        # both the field's own docstring ("expose only benchmark tools") and
        # the documented contract that /agent/analyze never places an order.
        # place_trade must only ever be reachable when benchmark_tools=True.
        response = self.client.post(
            "/agent/analyze",
            json={"question": "Will it rain tomorrow?", "tool_loop": True, "max_tool_steps": 1},
        )
        self.assertEqual(response.status_code, 200)
        system_prompt = self.provider.calls[0][0]["content"]
        self.assertNotIn("place_trade(", system_prompt)
        self.assertNotIn("web_search(", system_prompt)
        self.assertNotIn("manage_notes(", system_prompt)
        self.assertIn("forecast(", system_prompt)
        self.assertIn("scan_markets(", system_prompt)

    def test_agent_analyze_tool_loop_routes_via_scads_alt_provider_for_model(self):
        # `req.model` (the field every other /agent/analyze and /predict path
        # uses for SCADS model selection) used to be silently ignored by
        # _agent_tool_loop -- it only consulted req.openrouter_model, so
        # passing model="minimax-m3" ran on the server's default model and
        # every model's agent_id collapsed onto the same shared identity.
        alt_provider = FakeProvider()
        alt_provider.response = {
            "thought": "remember this",
            "action": "manage_notes",
            "args": {"action": "add", "text": "Track Fed dates."},
        }
        with (
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"minimax-m3": "MiniMaxAI/MiniMax-M3-MXFP8"},
            ),
            mock.patch.object(server_module, "_scads_alt_provider", return_value=alt_provider) as alt_provider_mock,
            tempfile.TemporaryDirectory() as td,
            mock.patch.dict(
                os.environ,
                {"FORESEA_AGENT_NOTES_PATH": str(Path(td) / "notes.json")},
                clear=False,
            ),
        ):
            response = self.client.post(
                "/agent/analyze",
                json={
                    "question": "Will the Fed cut rates tomorrow?",
                    "tool_loop": True,
                    "benchmark_tools": True,
                    "model": "minimax-m3",
                    "max_tool_steps": 1,
                },
            )

            self.assertEqual(response.status_code, 200)
            alt_provider_mock.assert_called_once_with("minimax-m3")
            self.assertGreater(len(alt_provider.calls), 0)
            self.assertEqual(len(self.provider.calls), 0)  # server default was never used

            notes = json.loads((Path(td) / "notes.json").read_text())
        self.assertIn("minimax-m3", notes)
        self.assertNotIn("agent", notes)

    def test_agent_analyze_fetches_market_and_recommends(self):
        import analyzing_llm_rationale.market_data as md

        quote = {
            "platform": "Polymarket",
            "ident": "fed",
            "question": "Will the Fed cut rates before September 30, 2026?",
            "market_url": "https://polymarket.com/market/fed",
            "description": "Latest venue context.",
            "resolution_criteria": "Resolve from the official FOMC announcement.",
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
        self.assertEqual(report["live_trade_intent"]["platform"], "polymarket")
        self.assertEqual(report["live_trade_intent"]["ident"], "fed")
        self.assertEqual(report["live_trade_intent"]["outcome"], "yes")
        self.assertAlmostEqual(report["live_trade_intent"]["model_probability"], 0.70)
        self.assertAlmostEqual(report["live_trade_intent"]["market_probability"], 0.40)
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("Latest venue context.", prompt)
        self.assertIn("Resolve from the official FOMC announcement.", prompt)

    def test_live_trade_intent_complements_a_no_recommendation(self):
        intent = server_module._live_trade_intent(
            platform="Kalshi",
            ident="KXFED-26SEP-C",
            market_url="https://kalshi.com/markets/KXFED-26SEP-C",
            question_type="binary",
            recommendation="buy_no",
            model_probability=0.30,
            market_probability=0.50,
            edge=-0.20,
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.platform, "kalshi")
        self.assertEqual(intent.outcome, "no")
        self.assertAlmostEqual(intent.model_probability, 0.70)
        self.assertAlmostEqual(intent.market_probability, 0.50)
        self.assertAlmostEqual(intent.edge, 0.20)

    def test_agent_analyze_preserves_ui_rules_and_supplied_articles(self):
        response = self.client.post(
            "/agent/analyze",
            json={
                "question": "Will Project Atlas launch before December 31, 2026?",
                "market_platform": "Kalshi",
                "market_probability": 0.45,
                "description": "The venue's background for Project Atlas.",
                "resolution_criteria": "Resolve from the company's launch announcement.",
                "categories": ["Technology"],
                "news_articles": [{
                    "title": "Atlas enters final testing",
                    "source": "Example News",
                    "summary": "The project entered its final testing phase.",
                }],
            },
        )

        self.assertEqual(response.status_code, 200)
        prompt = self.provider.calls[0][-1]["content"]
        self.assertIn("The venue's background for Project Atlas.", prompt)
        self.assertIn("Resolve from the company's launch announcement.", prompt)
        self.assertIn("Atlas enters final testing", prompt)
        self.assertEqual(
            self.evidence_pipeline.calls,
            [("Will Project Atlas launch before December 31, 2026?", 20)],
        )
        self.assertIn("Central bank signals policy shift", prompt)

    def test_agent_analyze_requires_question_or_market(self):
        response = self.client.post("/agent/analyze", json={"evidence_top_k": 3})
        self.assertEqual(response.status_code, 422)

    def test_predict_requires_auth_when_api_key_configured(self):
        self._require_auth_patch.stop()
        try:
            with mock.patch.object(server_module, "_REQUIRED_API_KEY", "secret"):
                response = self.client.post(
                    "/predict",
                    json={
                        "question": "Will the Fed cut rates before July 31, 2026?",
                        "variant": "variant0_neutral_baseline",
                        "attach_evidence": False,
                    },
                )
            self.assertEqual(response.status_code, 401)
            self.assertIn("Authentication required", response.json()["detail"])
        finally:
            self._require_auth_mock = self._require_auth_patch.start()

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
        self.assertIn("Agent integration guide: https://foresea.ink/agents", r.text)
        self.assertIn("Agent manifest: https://foresea.ink/.well-known/agent.json", r.text)
        self.assertIn("Remote MCP server: https://foresea.ink/mcp/", r.text)
        self.assertIn("MCP discovery manifest", r.text)

    def test_llms_txt_describes_api(self):
        r = self.client.get("/llms.txt")
        self.assertEqual(r.status_code, 200)
        self.assertIn("# Foresea", r.text)
        self.assertIn("Agent integration guide", r.text)
        self.assertIn("/.well-known/agent.json", r.text)
        self.assertIn("Remote MCP server", r.text)
        self.assertIn("https://foresea.ink/mcp/", r.text)
        self.assertIn("foresea_forecast", r.text)
        self.assertIn("OpenClaw agents", r.text)
        self.assertIn("/predict/stream", r.text)
        self.assertIn("/predict", r.text)
        self.assertIn("/openapi.json", r.text)

    def test_agents_page(self):
        r = self.client.get("/agents")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers.get("content-type", ""))
        self.assertIn("Agent integration surface", r.text)
        self.assertIn("https://foresea.ink/mcp/", r.text)
        self.assertIn("OpenClaw agents", r.text)
        self.assertIn("/predict/stream", r.text)

    def test_page_routes_include_dynamic_context(self):
        cases = [
            ("/", "home", "no-cache"),
            ("/ask", "ask", "no-cache"),
            ("/edge", "edge", "no-cache"),
            ("/track", "track", "no-cache"),
            ("/ledger", "ledger", "no-cache"),
            ("/chat/conv_123", "chat", "no-cache"),
            ("/watchlist", "watchlist", "no-cache"),
            ("/trade", "trade", "no-cache"),
            ("/agents", "agents", "public, max-age=300"),
        ]
        for path, page, cache_control in cases:
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertIn("text/html", r.headers.get("content-type", ""))
                self.assertEqual(r.headers.get("cache-control"), cache_control)
                context = self._page_context(r)
                self.assertEqual(context["page"], page)
                self.assertEqual(context["path"], path)
                self.assertEqual(context["canonical"], f"https://foresea.ink{path}")
                self.assertEqual(context["api"]["radar"], "/radar")
                self.assertIn("</head>", r.text)

    def test_chat_page_route_does_not_reflect_hostile_path_input(self):
        r = self.client.get("/chat/conv_%22%3E%3Cimg%20src=x%20onerror=alert(1)%3E")

        self.assertEqual(r.status_code, 200)
        self.assertNotIn("<img src=x onerror=alert(1)>", r.text)
        self.assertIn("\\u003cimg", r.text)

    def test_chat_models_route_is_not_shadowed_by_chat_page_route(self):
        r = self.client.get("/chat/models")

        self.assertEqual(r.status_code, 200)
        self.assertIn("application/json", r.headers.get("content-type", ""))
        self.assertNotIn("<html", r.text.lower())

    def test_agent_manifest(self):
        r = self.client.get("/.well-known/agent.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "Foresea")
        self.assertEqual(body["mcp"]["endpoint"], "https://foresea.ink/mcp/")
        self.assertEqual(body["integrations"]["openclaw"]["mcp_config"]["mcpServers"]["foresea"]["url"], "https://foresea.ink/mcp/")
        self.assertEqual(body["http"]["streaming_forecast"]["path"], "/predict/stream")
        self.assertNotIn("foresea_radar", body["mcp"]["tools"])
        self.assertNotIn("radar", body["http"])

    def test_agent_manifest_alias(self):
        r = self.client.get("/agent.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["agent_integration_url"], "https://foresea.ink/agents")

    def test_ai_plugin_manifest(self):
        r = self.client.get("/.well-known/ai-plugin.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name_for_model"], "foresea")
        self.assertEqual(body["auth"]["type"], "none")
        self.assertEqual(body["api"]["url"], "https://foresea.ink/openapi.json")

    def test_mcp_discovery_manifest(self):
        r = self.client.get("/.well-known/mcp/server.json")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["name"], "ink.foresea/forecasting")
        self.assertEqual(body["remotes"][0]["type"], "streamable-http")
        self.assertEqual(body["remotes"][0]["url"], "https://foresea.ink/mcp/")
        self.assertIn("foresea_scan_markets", body["_meta"]["ink.foresea/tools"])

    def test_mcp_discovery_manifest_compat_alias(self):
        r = self.client.get("/.well-known/mcp.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["remotes"][0]["url"], "https://foresea.ink/mcp/")

    def test_sitemap_xml(self):
        r = self.client.get("/sitemap.xml")
        self.assertEqual(r.status_code, 200)
        self.assertIn("application/xml", r.headers.get("content-type", ""))
        self.assertIn("https://foresea.ink/", r.text)
        self.assertIn("https://foresea.ink/agents", r.text)

    def test_track_record_digest(self):
        r = self.client.get("/track-record/digest")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Foresea forecast track record", r.text)

    def test_benchmark_score(self):
        r = self.client.post("/benchmark/score", json={
            "label": "test-forecaster",
            "forecasts": [
                {"probability": 0.8, "outcome": 1, "market_probability": 0.6},
                {"probability": 0.3, "outcome": 0, "market_probability": 0.5},
                {"probability": 0.9, "outcome": 1, "market_probability": 0.7},
            ],
        })
        self.assertEqual(r.status_code, 200)
        b = r.json()
        self.assertEqual(b["n"], 3)
        self.assertEqual(b["label"], "test-forecaster")
        # All three calls correct -> accuracy 1.0; market provided -> skill present.
        self.assertEqual(b["accuracy"], 1.0)
        self.assertIn("skill_vs_market", b)
        self.assertIsNotNone(b["ece"])

    def test_benchmark_score_without_market(self):
        r = self.client.post("/benchmark/score", json={
            "forecasts": [{"probability": 0.7, "outcome": 1}],
        })
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("skill_vs_market", r.json())  # no market probs -> no skill

    def test_openapi_spec_is_public(self):
        # Agents introspect the API via the OpenAPI spec.
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["info"]["title"], "Foresea Intelligence API")

    def test_track_record_tick_endpoint_removed(self):
        # The compute-bearing tick now runs in a GitHub Action, not on Cloud Run.
        r = self.client.post("/track-record/tick")
        self.assertIn(r.status_code, (404, 405))

    def test_internal_forecast_evaluation_is_token_gated(self):
        import analyzing_llm_rationale.server as srv

        report = {
            "schema_version": 1,
            "generated_at": "2026-07-27T05:00:00+00:00",
            "model": "council",
            "cohorts": {
                "prospective_audit": {
                    "resolved_markets": 0,
                }
            },
            "promotion": {
                "status": "collecting",
                "eligible": False,
            },
        }
        with (
            mock.patch.object(srv, "_TRACK_RECORD_TOKEN", "tok"),
            mock.patch.object(
                srv,
                "_read_forecast_evaluation",
                return_value=report,
            ),
        ):
            unauthorized = self.client.get("/internal/forecast-evaluation")
            authorized = self.client.get(
                "/internal/forecast-evaluation",
                headers={"X-Track-Token": "tok"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(authorized.json()["promotion"]["status"], "collecting")
        self.assertIn("private", authorized.headers["cache-control"])
        self.assertNotIn(
            "/internal/forecast-evaluation",
            self.client.get("/openapi.json").json()["paths"],
        )

    def test_internal_forecast_evaluation_returns_404_before_generation(self):
        import analyzing_llm_rationale.server as srv

        with (
            mock.patch.object(srv, "_TRACK_RECORD_TOKEN", "tok"),
            mock.patch.object(
                srv,
                "_read_forecast_evaluation",
                return_value=None,
            ),
        ):
            response = self.client.get(
                "/internal/forecast-evaluation",
                headers={"X-Track-Token": "tok"},
            )

        self.assertEqual(response.status_code, 404)

    def test_track_record_serves_backtest_when_no_resolved_live(self):
        import analyzing_llm_rationale.server as srv
        # No resolved live forecasts → fall back to the static backtest.
        with mock.patch.object(srv, "_read_live_track_record",
                               return_value={"n_snapshots_resolved": 0}):
            r = self.client.get("/track-record")
        self.assertEqual(r.status_code, 200)
        self.assertIn("methodology", r.json())

    def test_track_record_serves_live_when_resolved(self):
        import analyzing_llm_rationale.server as srv
        live = {"source": "live", "n_snapshots_resolved": 5, "overall": {"accuracy": 0.6}}
        with mock.patch.object(srv, "_read_live_track_record", return_value=live):
            r = self.client.get("/track-record")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["source"], "live")

    def test_edge_board_returns_disagreements_and_calibration(self):
        import analyzing_llm_rationale.server as srv
        live = {
            "generated_at": "2026-06-06T00:00:00+00:00",
            "n_markets_open": 1,
            "n_snapshots_resolved": 20,
            "edge_board": [{"platform": "Polymarket", "edge": 0.3, "edge_bucket": "20pp+",
                            "track_record": {"skill_significant": True}}],
            "by_edge": [{"edge_bucket": "20pp+", "skill_vs_market": 0.05, "skill_significant": True}],
            "lead_lag": {"market_converged_to_model_pct": 0.7},
        }
        with (
            mock.patch.object(srv, "_read_live_track_record", return_value=live),
            mock.patch.object(srv, "_read_mark_to_market_record", return_value=None),
        ):
            r = self.client.get("/edge-board")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["edge_board"][0]["edge"], 0.3)
        self.assertTrue(body["by_edge"][0]["skill_significant"])
        self.assertEqual(body["lead_lag"]["market_converged_to_model_pct"], 0.7)

    def test_edge_board_empty_when_no_live_file(self):
        import analyzing_llm_rationale.server as srv
        with (
            mock.patch.object(srv, "_read_live_track_record", return_value=None),
            mock.patch.object(srv, "_read_mark_to_market_record", return_value=None),
        ):
            r = self.client.get("/edge-board")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["edge_board"], [])

    def test_market_history_and_explain_shift(self):
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path

        from analyzing_llm_rationale.trackrec_store import DuckDBStore, Entity

        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "test_track_record.duckdb"
        store = DuckDBStore(db_path)
        try:
            snap1 = Entity(store.key("ForecastSnapshot", "Polymarket:fed-sept-26:gpt-oss-120b:2026-06-03"))
            snap1.update(
                platform="Polymarket",
                ident="fed-sept-26",
                model="gpt-oss-120b",
                snapshot_date="2026-06-03",
                snapshot_ts=datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc),
                question="Will the Fed cut rates?",
                market_url="https://polymarket.com/fed-sept-26",
                model_probability=0.40,
                market_probability=0.42,
                rationale="Interest rate cuts are unlikely in June given high CPI.",
                resolved=False,
            )
            store.put(snap1)

            snap2 = Entity(store.key("ForecastSnapshot", "Polymarket:fed-sept-26:gpt-oss-120b:2026-06-04"))
            snap2.update(
                platform="Polymarket",
                ident="fed-sept-26",
                model="gpt-oss-120b",
                snapshot_date="2026-06-04",
                snapshot_ts=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
                question="Will the Fed cut rates?",
                market_url="https://polymarket.com/fed-sept-26",
                model_probability=0.55,
                market_probability=0.52,
                rationale="Retrieved news confirming CPI dropped, opening path for rate cut.",
                resolved=False,
            )
            store.put(snap2)
            store.save()
        finally:
            store.close()

        with (
            mock.patch.dict("os.environ", {"TRACK_STORE_PATH": str(db_path)}),
            mock.patch("analyzing_llm_rationale.gcs_store.ensure_local_copy", return_value=True),
        ):
            r_hist = self.client.get("/market/history", params={"platform": "polymarket", "ident": "fed-sept-26"})
            self.assertEqual(r_hist.status_code, 200)
            hist_data = r_hist.json()["history"]
            self.assertEqual(len(hist_data), 2)
            self.assertEqual(hist_data[0]["model_probability"], 0.40)
            self.assertEqual(hist_data[1]["model_probability"], 0.55)
            self.assertEqual(hist_data[1]["rationale"], "Retrieved news confirming CPI dropped, opening path for rate cut.")

            self.provider.response = "The model increased cuts probability to 55.0% due to CPI dropping."
            r_explain = self.client.post("/market/explain-shift", json={"platform": "polymarket", "ident": "fed-sept-26"})
            self.assertEqual(r_explain.status_code, 200)
            exp_data = r_explain.json()
            self.assertEqual(exp_data["latest_prob"], 0.55)
            self.assertEqual(exp_data["previous_prob"], 0.40)
            self.assertAlmostEqual(exp_data["shift"], 0.15)
            self.assertEqual(exp_data["explanation"], "The model increased cuts probability to 55.0% due to CPI dropping.")

        temp_dir.cleanup()

    def test_market_history_and_explain_shift_hydrate_from_markets_table(self):
        # Rows written after market-level fields stopped being duplicated per
        # snapshot carry question/market_url as NULL on the row itself --
        # confirm both endpoints still return the real values, joined in from
        # the normalized `markets` table.
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path

        from analyzing_llm_rationale.trackrec_store import DuckDBStore, Entity

        temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_dir.name) / "test_track_record.duckdb"
        store = DuckDBStore(db_path)
        try:
            market = Entity(store.key("Market", "Polymarket:fed-sept-26"))
            market.update(
                platform="Polymarket",
                ident="fed-sept-26",
                question="Will the Fed cut rates?",
                market_url="https://polymarket.com/fed-sept-26",
            )
            store.put(market)

            snap1 = Entity(store.key("ForecastSnapshot", "Polymarket:fed-sept-26:gpt-oss-120b:2026-06-03"))
            snap1.update(
                platform="Polymarket",
                ident="fed-sept-26",
                model="gpt-oss-120b",
                snapshot_date="2026-06-03",
                snapshot_ts=datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc),
                model_probability=0.40,
                market_probability=0.42,
                rationale="Interest rate cuts are unlikely in June given high CPI.",
                resolved=False,
            )
            store.put(snap1)

            snap2 = Entity(store.key("ForecastSnapshot", "Polymarket:fed-sept-26:gpt-oss-120b:2026-06-04"))
            snap2.update(
                platform="Polymarket",
                ident="fed-sept-26",
                model="gpt-oss-120b",
                snapshot_date="2026-06-04",
                snapshot_ts=datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc),
                model_probability=0.55,
                market_probability=0.52,
                rationale="Retrieved news confirming CPI dropped, opening path for rate cut.",
                resolved=False,
            )
            store.put(snap2)
            store.save()

            # The snapshots themselves never carry question/market_url directly.
            raw = store._con.execute(
                "SELECT question, market_url FROM forecast_snapshot WHERE key = ?",
                ["Polymarket:fed-sept-26:gpt-oss-120b:2026-06-03"],
            ).fetchone()
            self.assertEqual(raw, (None, None))
        finally:
            store.close()

        with (
            mock.patch.dict("os.environ", {"TRACK_STORE_PATH": str(db_path)}),
            mock.patch("analyzing_llm_rationale.gcs_store.ensure_local_copy", return_value=True),
        ):
            r_hist = self.client.get("/market/history", params={"platform": "polymarket", "ident": "fed-sept-26"})
            self.assertEqual(r_hist.status_code, 200)
            hist_data = r_hist.json()["history"]
            self.assertEqual(len(hist_data), 2)
            self.assertEqual(hist_data[0]["question"], "Will the Fed cut rates?")
            self.assertEqual(hist_data[0]["market_url"], "https://polymarket.com/fed-sept-26")
            self.assertEqual(hist_data[1]["question"], "Will the Fed cut rates?")

            self.provider.response = "The model raised its cut probability after fresh CPI data."
            r_explain = self.client.post("/market/explain-shift", json={"platform": "polymarket", "ident": "fed-sept-26"})
            self.assertEqual(r_explain.status_code, 200)
            exp_data = r_explain.json()
            self.assertEqual(exp_data["latest_prob"], 0.55)
            self.assertEqual(exp_data["previous_prob"], 0.40)
            # The LLM prompt itself must carry the real question, not "None" --
            # the exact regression this hydration fix addresses.
            sent_prompt = self.provider.calls[-1][-1]["content"]
            self.assertIn("Question: Will the Fed cut rates?", sent_prompt)

        temp_dir.cleanup()

    def test_market_history_is_rate_limited(self):
        # This endpoint didn't do any real I/O in production before the GCS
        # migration (Dockerfile never bundled data/, so it always hit the
        # store_path.exists() guard instantly) -- now ensure_local_copy makes
        # a live GCS call per request, so it must be rate limited like every
        # other endpoint that does real work.
        import analyzing_llm_rationale.server as srv

        with (
            mock.patch.object(srv._rate_limiter, "_calls", 1),
            mock.patch("analyzing_llm_rationale.gcs_store.ensure_local_copy", return_value=False),
        ):
            first = self.client.get("/market/history", params={"platform": "polymarket", "ident": "anything"})
            second = self.client.get("/market/history", params={"platform": "polymarket", "ident": "anything"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_crypto_5m_equity_endpoint_returns_candidate_curves(self):
        import analyzing_llm_rationale.server as srv
        payload = {
            "generated_at": "2026-06-17T00:00:00+00:00",
            "since_hours": 72,
            "resolved_rows": 10,
            "curves": [{
                "key": "btc_inverse_edge_004",
                "label": "BTC inverse edge >=4pp",
                "points": [0.5, 1.0],
                "trades": 2,
                "hit_rate": 1.0,
                "pnl_per_contract": 1.0,
            }],
        }
        with mock.patch.object(srv.crypto_5m, "crypto_5m_candidate_equity", return_value=payload) as fn:
            r = self.client.get("/crypto-5m/equity?hours=48")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["curves"][0]["key"], "btc_inverse_edge_004")
        fn.assert_called_once_with(since_hours=48.0)

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

    def test_predict_skips_cold_knowledge_base_for_signed_in_user(self):
        from analyzing_llm_rationale import rag

        rag._embedder = None
        rag._embedder_loaded = False
        headers = {"Authorization": "Bearer " + _issue_session("coldkb", "cold@e.com", "K", "")}
        with mock.patch.object(
            server_module,
            "_rag_search",
            side_effect=AssertionError("cold RAG should not run on forecast path"),
        ):
            response = self.client.post(
                "/predict",
                json={"question": "Will the Fed cut the rate this year?", "attach_evidence": False},
                headers=headers,
            )

        self.assertEqual(response.status_code, 200)
        sources = [s["source"] for s in response.json().get("evidence_sources", [])]
        self.assertNotIn("Knowledge base", sources)
        self.assertFalse(rag.is_loaded())

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
        if payload.get("by_day"):
            self.assertGreaterEqual(payload["by_day"][0]["visits"], 1)

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
                "chat_mode": False,
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
                "chat_mode": False,
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
                "chat_mode": False,
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
                "chat_mode": False,
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

    def test_authenticated_user_can_manage_a_personal_ledger(self):
        token = _issue_session("user-ledger", "ledger@example.com", "Ledger User", "")
        headers = {"Authorization": f"Bearer {token}"}
        entry = {
            "id": "ledger_conv_test_msg_assistant",
            "conversation_id": "conv_test",
            "message_id": "msg_assistant",
            "question": "Will the Fed cut rates before July 2026?",
            "predicted_answer": "YES",
            "probability": 0.65,
            "rationale": "Inflation is easing.",
            "model": "gpt-oss-120b",
            "createdAt": 1002,
        }

        save_response = self.client.put(
            "/personal-ledger/ledger_conv_test_msg_assistant", json=entry, headers=headers
        )
        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(save_response.json()["probability"], 0.65)

        list_response = self.client.get("/personal-ledger", headers=headers)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["entries"], [entry])

        delete_response = self.client.delete(
            "/personal-ledger/ledger_conv_test_msg_assistant", headers=headers
        )
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(self.client.get("/personal-ledger", headers=headers).json()["entries"], [])

    def test_personal_ledger_isolated_between_authenticated_users(self):
        owner_headers = {"Authorization": f"Bearer {_issue_session('ledger-owner', 'owner@example.com', 'Owner', '')}"}
        other_headers = {"Authorization": f"Bearer {_issue_session('ledger-other', 'other@example.com', 'Other', '')}"}
        entry = {
            "id": "ledger_private_forecast",
            "conversation_id": "conv_owner_only",
            "message_id": "msg_owner_only",
            "question": "Private forecast question",
            "predicted_answer": "YES",
            "probability": 0.72,
            "rationale": "Private rationale.",
            "model": "gpt-oss-120b",
            "createdAt": 1002,
        }
        self.assertEqual(
            self.client.put("/personal-ledger/ledger_private_forecast", json=entry, headers=owner_headers).status_code,
            200,
        )

        self.assertEqual(self.client.get("/personal-ledger", headers=other_headers).json()["entries"], [])
        self.assertEqual(
            self.client.delete("/personal-ledger/ledger_private_forecast", headers=other_headers).status_code,
            200,
        )
        self.assertEqual(
            self.client.get("/personal-ledger", headers=owner_headers).json()["entries"], [entry]
        )

    def test_personal_ledger_rejects_tampered_or_invalid_entries(self):
        headers = {"Authorization": f"Bearer {_issue_session('ledger-validate', 'validate@example.com', 'Validate', '')}"}
        entry = {
            "id": "ledger_valid_entry",
            "conversation_id": "conv_test",
            "message_id": "msg_test",
            "question": "Will the test pass?",
            "predicted_answer": "YES",
            "probability": 0.65,
            "rationale": "Test rationale.",
            "model": "gpt-oss-120b",
            "createdAt": 1002,
        }

        mismatch = dict(entry, id="ledger_other_entry")
        response = self.client.put("/personal-ledger/ledger_valid_entry", json=mismatch, headers=headers)
        self.assertEqual(response.status_code, 400)

        for probability in (-0.01, 1.01):
            response = self.client.put(
                "/personal-ledger/ledger_valid_entry",
                json=dict(entry, probability=probability),
                headers=headers,
            )
            self.assertEqual(response.status_code, 422)

        response = self.client.put(
            "/personal-ledger/ledger_valid_entry",
            json=dict(entry, question="x" * 801),
            headers=headers,
        )
        self.assertEqual(response.status_code, 422)

        response = self.client.put(
            "/personal-ledger/ledger_valid_entry",
            json=dict(entry, rationale="x" * 8001),
            headers=headers,
        )
        self.assertEqual(response.status_code, 422)

    def test_personal_ledger_retries_are_idempotent(self):
        headers = {"Authorization": f"Bearer {_issue_session('ledger-retry', 'retry@example.com', 'Retry', '')}"}
        entry = {
            "id": "ledger_retry_entry",
            "conversation_id": "conv_retry",
            "message_id": "msg_retry",
            "question": "Will retry deduplicate?",
            "predicted_answer": "YES",
            "probability": 0.51,
            "rationale": "First save.",
            "model": "gpt-oss-120b",
            "createdAt": 1002,
        }
        self.assertEqual(
            self.client.put("/personal-ledger/ledger_retry_entry", json=entry, headers=headers).status_code,
            200,
        )
        retried = dict(entry, probability=0.61, rationale="Retry save.")
        self.assertEqual(
            self.client.put("/personal-ledger/ledger_retry_entry", json=retried, headers=headers).status_code,
            200,
        )
        self.assertEqual(self.client.get("/personal-ledger", headers=headers).json()["entries"], [retried])

    def test_personal_ledger_feedback_is_private_validated_and_durable(self):
        owner_headers = {"Authorization": f"Bearer {_issue_session('ledger-feedback-owner', 'owner@example.com', 'Owner', '')}"}
        other_headers = {"Authorization": f"Bearer {_issue_session('ledger-feedback-other', 'other@example.com', 'Other', '')}"}
        entry = {
            "id": "ledger_feedback_entry",
            "conversation_id": "conv_feedback",
            "message_id": "msg_feedback",
            "question": "Will feedback remain private?",
            "predicted_answer": "YES",
            "probability": 0.73,
            "rationale": "The entry belongs only to its owner.",
            "model": "gpt-oss-120b",
            "createdAt": 1002,
        }
        self.assertEqual(
            self.client.put("/personal-ledger/ledger_feedback_entry", json=entry, headers=owner_headers).status_code,
            200,
        )

        invalid = self.client.patch(
            "/personal-ledger/ledger_feedback_entry/verdict",
            json={"verdict": "maybe"},
            headers=owner_headers,
        )
        self.assertEqual(invalid.status_code, 422)

        private = self.client.patch(
            "/personal-ledger/ledger_feedback_entry/verdict",
            json={"verdict": "wrong"},
            headers=other_headers,
        )
        self.assertEqual(private.status_code, 404)

        correct = self.client.patch(
            "/personal-ledger/ledger_feedback_entry/verdict",
            json={"verdict": "correct"},
            headers=owner_headers,
        )
        self.assertEqual(correct.status_code, 200)
        self.assertEqual(correct.json()["user_verdict"], "correct")
        self.assertGreaterEqual(correct.json()["judgedAt"], entry["createdAt"])

        # A duplicate add from the chat card must not erase a verdict the user set.
        retry = self.client.put(
            "/personal-ledger/ledger_feedback_entry",
            json=dict(entry, rationale="Retry did not erase feedback."),
            headers=owner_headers,
        )
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(retry.json()["user_verdict"], "correct")

        changed = self.client.patch(
            "/personal-ledger/ledger_feedback_entry/verdict",
            json={"verdict": "wrong"},
            headers=owner_headers,
        )
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(changed.json()["user_verdict"], "wrong")

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
        with mock.patch.object(server_module, "_require_auth", return_value={"sub": "api-key-user"}):
            first = self.client.post("/predict", json=payload)
            second = self.client.post("/predict", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        # The model provider is invoked only once; the second call hits the cache.
        self.assertEqual(len(self.provider.calls), 1)

    def test_predict_cache_is_scoped_by_signed_in_user(self):
        payload = {
            "question": "Will the Fed cut rates before December 31, 2026?",
            "question_type": "binary",
            "attach_evidence": False,
        }
        first_user = {"Authorization": "Bearer " + _issue_session("user-a", "a@example.com", "A", "")}
        second_user = {"Authorization": "Bearer " + _issue_session("user-b", "b@example.com", "B", "")}

        first = self.client.post("/predict", json=payload, headers=first_user)
        second = self.client.post("/predict", json=payload, headers=second_user)
        repeat_first = self.client.post("/predict", json=payload, headers=first_user)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(repeat_first.status_code, 200)
        self.assertEqual(first.json(), repeat_first.json())
        self.assertEqual(len(self.provider.calls), 2)

    def test_chat_predict_uses_fast_interactive_default_model(self):
        fast_provider = FakeProvider()
        with (
            mock.patch.object(server_module, "_INTERACTIVE_DEFAULT_MODEL", "minimax-m3"),
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"minimax-m3": "MiniMaxAI/MiniMax-M3-MXFP8"},
            ),
            mock.patch.object(server_module, "_scads_alt_provider", return_value=fast_provider) as alt_provider,
            mock.patch.dict(os.environ, {"SCADS_AI_API_KEY": "test-key"}, clear=False),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "chat_mode": True,
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_key"], "minimax-m3")
        alt_provider.assert_called_once_with("minimax-m3")
        self.assertEqual(len(fast_provider.calls), 1)
        self.assertEqual(len(self.provider.calls), 0)

    def test_typed_predict_keeps_server_default_model(self):
        with (
            mock.patch.object(server_module, "_INTERACTIVE_DEFAULT_MODEL", "minimax-m3"),
            mock.patch.object(
                server_module,
                "_SCADS_MODEL_ALLOWLIST",
                {"minimax-m3": "MiniMaxAI/MiniMax-M3-MXFP8"},
            ),
            mock.patch.object(server_module, "_scads_alt_provider") as alt_provider,
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "chat_mode": False,
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_key"], "test-model")
        alt_provider.assert_not_called()
        self.assertEqual(len(self.provider.calls), 1)

    def test_chat_fallback_uses_one_bounded_primary_attempt(self):
        import asyncio

        primary = FailingProvider("primary-model")
        fallback = FakeProvider()
        req = server_module.PredictRequest(
            question="Will the Fed cut rates before July 31, 2026?",
            chat_mode=True,
            attach_evidence=False,
        )

        with (
            mock.patch.object(server_module, "_CHAT_PROVIDER_TIMEOUT_S", 0.01),
            mock.patch.object(server_module, "_CHAT_PROVIDER_MAX_RETRIES", 0),
            mock.patch.object(server_module, "_SCADS_MODEL_FALLBACKS", {"test-model": ("fallback-model",)}),
            mock.patch.object(server_module, "_scads_provider_for_model_name", return_value=fallback),
        ):
            content, served = asyncio.run(
                server_module._provider_chat_with_chat_fallbacks(
                    req,
                    primary,
                    [{"role": "user", "content": "Question"}],
                    0.0,
                    128,
                )
            )

        self.assertEqual(primary.calls, 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertIs(served, fallback)
        self.assertIn("predicted_answer", content)

    def test_chat_provider_default_keeps_transient_retry_enabled(self):
        self.assertEqual(server_module._CHAT_PROVIDER_MAX_RETRIES, 1)

    def test_stream_chat_fallback_after_first_token_timeout(self):
        import asyncio

        primary = SlowStreamProvider()
        fallback = FakeProvider()
        fallback.stream_response = "fast fallback"
        req = server_module.PredictRequest(
            question="Will the Fed cut rates before July 31, 2026?",
            chat_mode=True,
            attach_evidence=False,
        )

        async def collect_chunks():
            used = {"provider": primary}
            chunks = []
            async for chunk in server_module._provider_stream_chat_with_chat_fallbacks(
                req,
                primary,
                [{"role": "user", "content": "Question"}],
                0.0,
                128,
                used,
            ):
                chunks.append(chunk)
            return chunks, used["provider"]

        with (
            mock.patch.object(server_module, "_CHAT_PROVIDER_TIMEOUT_S", 0.01),
            mock.patch.object(server_module, "_SCADS_MODEL_FALLBACKS", {"test-model": ("fallback-model",)}),
            mock.patch.object(server_module, "_scads_provider_for_model_name", return_value=fallback),
        ):
            chunks, served = asyncio.run(collect_chunks())

        self.assertEqual(primary.calls, 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertIs(served, fallback)
        self.assertEqual("".join(chunks), "fast fallback")

    def test_chat_predict_uses_interactive_token_cap(self):
        with (
            mock.patch.object(server_module, "_INTERACTIVE_DEFAULT_MODEL", ""),
            mock.patch.object(server_module, "_INTERACTIVE_MAX_TOKENS", 128),
        ):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "chat_mode": True,
                    "attach_evidence": False,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.provider.max_tokens, [128])

    def test_predict_respects_request_max_tokens_with_server_cap(self):
        _state["max_tokens"] = 256
        with mock.patch.object(server_module, "_INTERACTIVE_DEFAULT_MODEL", ""):
            response = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before July 31, 2026?",
                    "chat_mode": True,
                    "attach_evidence": False,
                    "max_tokens": 128,
                },
            )
            capped = self.client.post(
                "/predict",
                json={
                    "question": "Will the Fed cut rates before August 31, 2026?",
                    "chat_mode": True,
                    "attach_evidence": False,
                    "max_tokens": 1024,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(capped.status_code, 200)
        self.assertEqual(self.provider.max_tokens, [128, 256])

    def test_predict_cache_isolated_by_requested_model(self):
        payload = {
            "question": "Will the Fed cut rates before November 30, 2026?",
            "question_type": "binary",
            "attach_evidence": False,
            "chat_mode": False,
        }
        first = self.client.post("/predict", json=payload)
        with mock.patch.object(server_module, "_SCADS_MODEL_ALLOWLIST", {"test-model": {}}):
            council = self.client.post("/predict", json={**payload, "model": "council"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(council.status_code, 200)
        self.assertEqual(first.json()["model_key"], "test-model")
        self.assertEqual(council.json()["model_key"], "council")
        self.assertIn("[Council debate]", council.json()["rationale"])

    def test_predict_cache_includes_resolution_criteria(self):
        payload = {
            "question": "Will Project Atlas launch in 2026?",
            "question_type": "binary",
            "attach_evidence": False,
            "chat_mode": False,
        }
        first = self.client.post(
            "/predict",
            json={**payload, "resolution_criteria": "Resolve from the company announcement."},
        )
        second = self.client.post(
            "/predict",
            json={**payload, "resolution_criteria": "Resolve from the regulator filing."},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(self.provider.calls), 2)

    def test_short_followup_fetches_contextualized_fresh_evidence(self):
        # A short follow-up in a thread is too terse for standalone retrieval,
        # but Foresea should still fetch raw evidence anchored to thread context.
        response = self.client.post(
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
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.evidence_pipeline.calls), 1)
        query, top_k = self.evidence_pipeline.calls[0]
        self.assertIn("Who wins AL vs WE in the LPL series?", query)
        self.assertIn("Follow-up: WE is 90+", query)
        self.assertEqual(top_k, 20)

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
