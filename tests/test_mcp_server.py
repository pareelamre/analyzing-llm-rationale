from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import mcp_server as mcp  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = text

    def json(self):
        if isinstance(self._payload, ValueError):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers or {},
            "json": json,
            "params": params,
            "timeout": timeout,
        })
        return self.responses.pop(0)


class FakeAsyncSession(FakeSession):
    async def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        return super().request(method, url, headers=headers, json=json, params=params, timeout=timeout)


class PayloadTests(unittest.TestCase):
    def test_predict_payload_maps_optional_market_fields(self):
        payload = mcp.build_predict_payload(
            question="Will the Fed cut rates before September 30, 2026?",
            question_type="binary",
            market_platform="Polymarket",
            market_probability=42,
            attach_evidence=False,
            options=[],
            categories=["Economics"],
        )

        self.assertEqual(payload["question_type"], "binary")
        self.assertEqual(payload["market_platform"], "Polymarket")
        self.assertEqual(payload["market_probability"], 42)
        self.assertEqual(payload["attach_evidence"], False)
        self.assertNotIn("options", payload)
        self.assertEqual(payload["categories"], ["Economics"])

    def test_agent_payload_omits_absent_identifier_fields(self):
        payload = mcp.build_agent_analyze_payload(
            question="Will X happen?",
            platform="polymarket",
            slug="will-x-happen",
            builtin_skills=True,
        )

        self.assertEqual(payload["slug"], "will-x-happen")
        self.assertEqual(payload["builtin_skills"], True)
        self.assertNotIn("ticker", payload)
        self.assertNotIn("market_id", payload)


class ForeseaClientTests(unittest.TestCase):
    def test_forecast_posts_predict_with_api_key(self):
        session = FakeSession(FakeResponse(payload={"predicted_answer": "Yes"}))
        client = mcp.ForeseaClient(
            base_url="https://foresea.test/",
            api_key="secret",
            timeout_s=7,
            session=session,
        )

        result = client.forecast({"question": "Will X happen?"})

        self.assertEqual(result, {"predicted_answer": "Yes"})
        call = session.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://foresea.test/predict")
        self.assertEqual(call["headers"]["X-API-Key"], "secret")
        self.assertEqual(call["json"], {"question": "Will X happen?"})
        self.assertEqual(call["timeout"], 7)

    def test_scan_markets_uses_query_params(self):
        session = FakeSession(FakeResponse(payload={"platform": "Polymarket", "opportunities": []}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        result = client.scan_markets(platform="all", limit=2, min_edge=0.2, query="fed")

        self.assertEqual(result["platform"], "Polymarket")
        call = session.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://foresea.test/agent/scan")
        self.assertEqual(call["params"], {
            "platform": "all",
            "limit": 2,
            "min_edge": 0.2,
            "evidence_top_k": 3,
            "query": "fed",
        })

    def test_batch_quotes_sends_refs_as_repeated_query_params(self):
        session = FakeSession(FakeResponse(payload={"quotes": [], "count": 0}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        result = client.batch_quotes(["kalshi:KXFED-25JUN-H", "polymarket:fed-cut-2026"])

        self.assertEqual(result, {"quotes": [], "count": 0})
        call = session.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://foresea.test/market/batch")
        self.assertEqual(call["params"], {"refs": ["kalshi:KXFED-25JUN-H", "polymarket:fed-cut-2026"]})

    def test_check_run_returns_running_status_without_fetching_detail(self):
        session = FakeSession(FakeResponse(payload={
            "runs": [{"id": "agent_run_abc", "status": "running", "client_run_key": "key1"}],
        }))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        result = client.check_run("key1")

        self.assertEqual(result, {"status": "running", "id": "agent_run_abc", "detail": "Still in progress -- check back again shortly."})
        self.assertEqual(len(session.calls), 1)  # no second request for detail
        self.assertEqual(session.calls[0]["params"], {"client_run_key": "key1"})

    def test_check_run_fetches_full_detail_once_the_run_is_complete(self):
        session = FakeSession(
            FakeResponse(payload={"runs": [{"id": "agent_run_abc", "status": "completed", "client_run_key": "key1"}]}),
            FakeResponse(payload={"report": {"predicted_answer": "Yes"}}),
        )
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        result = client.check_run("key1")

        self.assertEqual(result, {"report": {"predicted_answer": "Yes"}})
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1]["url"], "https://foresea.test/agent/runs/agent_run_abc")

    def test_check_run_raises_404_when_no_run_matches_the_key(self):
        session = FakeSession(FakeResponse(payload={"runs": []}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        with self.assertRaises(mcp.ForeseaApiError) as ctx:
            client.check_run("no-such-key")

        self.assertEqual(ctx.exception.status_code, 404)

    def test_http_error_parses_detail(self):
        session = FakeSession(FakeResponse(status_code=422, payload={"detail": "bad request"}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        with self.assertRaises(mcp.ForeseaApiError) as ctx:
            client.track_record()

        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail, "bad request")

    def test_non_json_response_is_error(self):
        session = FakeSession(FakeResponse(payload=ValueError("no json"), text="not json"))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)

        with self.assertRaises(mcp.ForeseaApiError) as ctx:
            client.openapi()

        self.assertIn("non-JSON", ctx.exception.detail)


class ForeseaAsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_forecast_posts_predict_without_blocking_server_loop(self):
        session = FakeAsyncSession(FakeResponse(payload={"predicted_answer": "No"}))
        client = mcp.ForeseaClient(
            base_url="https://foresea.test/",
            api_key="secret",
            timeout_s=11,
            async_session=session,
        )

        result = await client.aforecast({"question": "Will Y happen?"})

        self.assertEqual(result, {"predicted_answer": "No"})
        call = session.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["url"], "https://foresea.test/predict")
        self.assertEqual(call["headers"]["X-API-Key"], "secret")
        self.assertEqual(call["json"], {"question": "Will Y happen?"})
        self.assertEqual(call["timeout"], 11)

    async def test_async_batch_quotes_sends_refs_as_repeated_query_params(self):
        session = FakeAsyncSession(FakeResponse(payload={"quotes": [{"platform": "kalshi", "ident": "T"}], "count": 1}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        result = await client.abatch_quotes(["kalshi:T"])

        self.assertEqual(result["count"], 1)
        call = session.calls[0]
        self.assertEqual(call["url"], "https://foresea.test/market/batch")
        self.assertEqual(call["params"], {"refs": ["kalshi:T"]})

    async def test_async_analyze_resilient_stamps_a_client_run_key(self):
        session = FakeAsyncSession(FakeResponse(payload={"predicted_answer": "Yes"}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        await client.aanalyze_resilient({"question": "Will X happen?"})

        sent_key = session.calls[0]["json"]["client_run_key"]
        self.assertTrue(sent_key)  # a uuid4 hex was generated

    async def test_async_analyze_resilient_preserves_a_caller_supplied_client_run_key(self):
        session = FakeAsyncSession(FakeResponse(payload={"predicted_answer": "Yes"}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        await client.aanalyze_resilient({"question": "Will X happen?", "client_run_key": "caller-chosen-key"})

        self.assertEqual(session.calls[0]["json"]["client_run_key"], "caller-chosen-key")

    async def test_async_analyze_resilient_names_the_run_key_on_timeout(self):
        import httpx

        class TimeoutSession(FakeAsyncSession):
            async def request(self, *args, **kwargs):
                raise httpx.TimeoutException("read timed out")

        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=TimeoutSession())

        with self.assertRaises(mcp.ForeseaApiError) as ctx:
            await client.aanalyze_resilient({"question": "Will X happen?", "client_run_key": "my-key"})

        self.assertEqual(ctx.exception.status_code, 504)
        self.assertIn("my-key", ctx.exception.detail)
        self.assertIn("foresea_check_run", ctx.exception.detail)

    async def test_async_check_run_returns_running_status_without_fetching_detail(self):
        session = FakeAsyncSession(FakeResponse(payload={
            "runs": [{"id": "agent_run_abc", "status": "running", "client_run_key": "key1"}],
        }))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        result = await client.acheck_run("key1")

        self.assertEqual(result["status"], "running")
        self.assertEqual(len(session.calls), 1)

    async def test_async_check_run_fetches_full_detail_once_complete(self):
        session = FakeAsyncSession(
            FakeResponse(payload={"runs": [{"id": "agent_run_abc", "status": "completed"}]}),
            FakeResponse(payload={"report": {"predicted_answer": "Yes"}}),
        )
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        result = await client.acheck_run("key1")

        self.assertEqual(result, {"report": {"predicted_answer": "Yes"}})
        self.assertEqual(session.calls[1]["url"], "https://foresea.test/agent/runs/agent_run_abc")

    async def test_async_http_error_parses_detail(self):
        session = FakeAsyncSession(FakeResponse(status_code=503, payload={"detail": "temporarily unavailable"}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)

        with self.assertRaises(mcp.ForeseaApiError) as ctx:
            await client.atrack_record()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "temporarily unavailable")

    async def test_async_exchange_status_returns_status_and_schedule(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.aexchange_status()
        self.assertIn("status", res)
        self.assertIn("schedule", res)

    async def test_async_market_tags_returns_list(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.amarket_tags()
        self.assertIsInstance(res, list)

    async def test_async_live_data_returns_dict(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.alive_data()
        self.assertIsInstance(res, dict)

    async def test_async_polymarket_meta_returns_list(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.apolymarket_meta(target="series")
        self.assertIsInstance(res, list)

    async def test_async_recent_trades_returns_list(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.arecent_trades(platform="kalshi", ticker_or_token="KXFED-25JUN-H")
        self.assertIsInstance(res, list)

    async def test_async_market_leaderboard_returns_list(self):
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        res = await client.amarket_leaderboard(limit=5)
        self.assertIsInstance(res, list)

if __name__ == "__main__":
    unittest.main()
