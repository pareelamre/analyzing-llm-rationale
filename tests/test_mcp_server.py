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

    async def test_async_venue_tools_return_real_adapter_shapes(self):
        from unittest.mock import patch

        from analyzing_llm_rationale import market_data
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        cases = [
            (client.amarket_tags, (), "fetch_polymarket_tags", [{"id": 1}]),
            (client.alive_data, ("KXBTC-TEST",), "fetch_kalshi_live_data", {"live_data": {}}),
            (client.apolymarket_meta, ("series",), "fetch_polymarket_series", [{"id": 2}]),
            (client.arecent_trades, ("kalshi", "KXFED-25JUN-H"), "fetch_recent_trades", [{"ticker": "KXFED-25JUN-H"}]),
            (client.amarket_leaderboard, (5,), "fetch_trader_leaderboard", [{"rank": "1"}]),
        ]
        for call, args, helper, payload in cases:
            with self.subTest(helper=helper), patch.object(market_data, helper, return_value=payload):
                self.assertEqual(await call(*args), payload)
        with patch.object(market_data, "fetch_kalshi_exchange_status", return_value={"exchange_active": True}), patch.object(
            market_data, "fetch_kalshi_exchange_schedule", return_value={"schedule": []},
        ):
            self.assertEqual(await client.aexchange_status(), {"status": {"exchange_active": True}, "schedule": {"schedule": []}})

    def test_weather_radar_queries_market_weather_radar(self):
        session = FakeSession(FakeResponse(payload={"opportunities": []}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", session=session)
        res = client.weather_radar(target_date="2026-09-07")
        self.assertEqual(res, {"opportunities": []})
        call = session.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://foresea.test/market/weather-radar")
        self.assertEqual(call["params"], {"target_date": "2026-09-07"})

    async def test_async_weather_radar_queries_market_weather_radar(self):
        session = FakeAsyncSession(FakeResponse(payload={"opportunities": [{"ident": "KXHIGHNY"}]}))
        client = mcp.ForeseaClient(base_url="https://foresea.test", async_session=session)
        res = await client.aweather_radar()
        self.assertEqual(res, {"opportunities": [{"ident": "KXHIGHNY"}]})
        call = session.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "https://foresea.test/market/weather-radar")

    def test_weather_forecast_calls_weather_research(self):
        from unittest.mock import patch

        from analyzing_llm_rationale import weather_research
        client = mcp.ForeseaClient(base_url="https://foresea.test")
        fake_research = {"station": "KNYC", "projected_high_f": 75.0}
        with patch.object(weather_research, "research_weather_market", return_value=fake_research) as mock_res:
            res = client.weather_forecast("KNYC", target_date="2026-09-07")
            self.assertEqual(res, fake_research)
            mock_res.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class TrackRecordSummaryTests(unittest.TestCase):
    """foresea_track_record could not answer the question it exists for.

    GET /track-record returns the full aggregate -- ~1.96M characters, of
    which models_comparison, paper_pnl and primary_paper_pnl are 98.4%. That
    exceeds an MCP client's response limit, so the call fails outright
    rather than reporting accuracy. The fields the tool's own description
    promises are 5,363 characters: 0.27% of what it sent.
    """

    def _payload(self):
        return {
            "generated_at": "2026-09-07T04:28:55+00:00",
            "overall": {"accuracy": 0.8116, "model_brier": 0.1415, "n": 2835},
            "by_horizon": [{"horizon": "7-14d", "n": 398}],
            "calibration": [{"bucket": 0.5, "n": 10}],
            "methodology": "resolved markets only",
            "models_comparison": ["x"] * 5000,
            "paper_pnl": {"rows": ["y"] * 5000},
            "primary_paper_pnl": {"rows": ["z"] * 5000},
        }

    def test_the_bulk_blocks_go_and_the_summary_stays(self):
        out = mcp._summarise_track_record(self._payload())
        for gone in ("models_comparison", "paper_pnl", "primary_paper_pnl"):
            self.assertNotIn(gone, out)
        for kept in ("overall", "by_horizon", "calibration", "methodology", "generated_at"):
            self.assertIn(kept, out)
        self.assertEqual(out["overall"]["accuracy"], 0.8116)

    def test_it_says_what_it_dropped_and_where_to_get_it(self):
        out = mcp._summarise_track_record(self._payload())
        omitted = out["omitted_for_size"]
        self.assertEqual(
            sorted(omitted["keys"]),
            ["models_comparison", "paper_pnl", "primary_paper_pnl"],
        )
        self.assertIn("/track-record", omitted["detail"])

    def test_it_shrinks_the_payload_by_orders_of_magnitude(self):
        import json as _json

        payload = self._payload()
        before = len(_json.dumps(payload))
        after = len(_json.dumps(mcp._summarise_track_record(payload)))
        self.assertLess(after * 10, before, "summary should be far smaller than the aggregate")

    def test_a_payload_without_bulk_blocks_is_untouched(self):
        lean = {"overall": {"accuracy": 1.0}, "generated_at": "t"}
        out = mcp._summarise_track_record(lean)
        self.assertEqual(out, lean)
        self.assertNotIn("omitted_for_size", out)

    def test_a_non_dict_passes_straight_through(self):
        for value in (None, [], "text", 3):
            with self.subTest(value=value):
                self.assertEqual(mcp._summarise_track_record(value), value)


class EdgeBoardSummaryTests(unittest.TestCase):
    """foresea_edge_board fails the same way, from the same aggregate.

    881,216 characters, also over an MCP client's limit. The three
    track-record bulk keys are 59.9% of it and the per-model ledger blocks
    another 28%, while ``edge_board`` -- the ranked markets the tool is
    named for -- is 9.5%.
    """

    def _payload(self):
        return {
            "generated_at": "2026-09-07T04:28:55+00:00",
            "edge_board": [{"ticker": f"M{i}", "edge": 0.1} for i in range(25)],
            "by_edge": [{"edge_bucket": "10-20pp", "n": 82}],
            "mark_to_market_account": {"value": 1.0},
            "models_comparison": ["x"] * 2000,
            "paper_pnl": {"rows": ["y"] * 2000},
            "primary_paper_pnl": {"rows": ["z"] * 2000},
            "mark_to_market_by_model": {"m": ["a"] * 2000},
            "quarter_kelly_by_model": {"m": ["b"] * 1000},
            "growth_1pct_by_model": {"m": ["c"] * 1000},
            "growth_2pct_by_model": {"m": ["d"] * 1000},
        }

    def test_the_ranked_markets_survive_and_the_ledgers_go(self):
        out = mcp._summarise_track_record(self._payload())
        self.assertEqual(len(out["edge_board"]), 25)
        self.assertIn("by_edge", out)
        self.assertIn("mark_to_market_account", out)
        for gone in ("models_comparison", "paper_pnl", "primary_paper_pnl",
                     "mark_to_market_by_model", "quarter_kelly_by_model",
                     "growth_1pct_by_model", "growth_2pct_by_model"):
            self.assertNotIn(gone, out)

    def test_only_keys_actually_present_are_reported_as_omitted(self):
        """track_record carries three of these; edge_board carries all seven."""
        lean = {"overall": {"n": 1}, "paper_pnl": {"rows": []}}
        out = mcp._summarise_track_record(lean)
        self.assertEqual(out["omitted_for_size"]["keys"], ["paper_pnl"])
class FeedLatestFallbackTests(unittest.TestCase):
    """The fallback crashed every time it ran.

        "market_edge_signals": self.edge_board()[:limit]

    edge_board() returns the aggregate mapping, and slicing a dict raises
    TypeError: unhashable type: 'slice'. So whenever /feed/latest was
    unavailable -- the only case this branch exists for -- the tool raised
    instead of degrading. Calling foresea_feed_latest reproduced it exactly.
    """

    class _GetSession(FakeSession):
        """feed_latest is the only client method that calls session.get()
        directly rather than going through _request(), so the shared
        FakeSession -- which implements request() only -- cannot drive it."""

        def get(self, url, params=None, timeout=None):
            self.calls.append({"method": "GET", "url": url, "params": params})
            return self.responses.pop(0)

    def _client(self, *responses):
        return mcp.ForeseaClient(
            base_url="https://foresea.test", session=self._GetSession(*responses)
        )

    def test_the_fallback_returns_ranked_markets_instead_of_raising(self):
        board = {"edge_board": [{"ticker": f"M{i}"} for i in range(5)], "paper_pnl": {}}
        client = self._client(FakeResponse(status_code=503), FakeResponse(payload=board))

        out = client.feed_latest(limit=3)

        self.assertEqual(len(out["market_edge_signals"]), 3)
        self.assertEqual(out["market_edge_signals"][0]["ticker"], "M0")
        self.assertIn("channels", out)

    def test_a_board_without_the_key_degrades_to_empty_not_an_error(self):
        client = self._client(FakeResponse(status_code=503), FakeResponse(payload={"other": 1}))
        self.assertEqual(client.feed_latest(limit=3)["market_edge_signals"], [])

    def test_a_non_object_board_response_surfaces_as_an_api_error(self):
        """_request rejects non-object JSON, so edge_board() cannot return a
        list. The isinstance guard in the fallback is belt-and-braces; the
        real contract is that this raises rather than degrading silently."""
        client = self._client(FakeResponse(status_code=503), FakeResponse(payload=["a"]))
        with self.assertRaises(mcp.ForeseaApiError):
            client.feed_latest(limit=2)

    def test_the_primary_path_is_still_preferred(self):
        client = self._client(FakeResponse(payload={"timestamp": "t", "signals": []}))
        self.assertEqual(client.feed_latest()["timestamp"], "t")


class EdgeBoardRowsTests(unittest.TestCase):
    """Two tools mistook the aggregate mapping for the list of opportunities.

    feed_latest sliced it (TypeError: unhashable type: 'slice');
    optimize_portfolio fed it to audit_edge_board, which iterates and gets
    the mapping's string keys (ValueError: dictionary update sequence element
    #0 has length 1; 2 is required). The second was hidden behind
    `except Exception as exc: return {"error": str(exc)}`, so it returned a
    plausible error object rather than a portfolio and never crashed.
    """

    def test_it_pulls_the_rows_out_of_the_aggregate(self):
        board = {"edge_board": [{"ticker": "M1"}, {"ticker": "M2"}], "paper_pnl": {}}
        self.assertEqual(len(mcp._edge_board_rows(board)), 2)

    def test_a_bare_list_passes_through(self):
        rows = [{"ticker": "M1"}]
        self.assertEqual(mcp._edge_board_rows(rows), rows)

    def test_anything_unusable_becomes_an_empty_list_not_an_exception(self):
        for board in ({}, {"edge_board": None}, {"edge_board": {}}, None, "text", 7):
            with self.subTest(board=board):
                self.assertEqual(mcp._edge_board_rows(board), [])

    def test_optimize_portfolio_no_longer_hands_a_mapping_to_the_auditor(self):
        """The real failure: audit_edge_board iterates what it is given."""
        from analyzing_llm_rationale.edge_credibility import audit_edge_board

        aggregate = {"edge_board": [{"ticker": "M1", "edge": 0.2}], "paper_pnl": {}}
        with self.assertRaises(ValueError):
            audit_edge_board(aggregate)
        self.assertEqual(
            audit_edge_board(mcp._edge_board_rows(aggregate))[0]["ticker"], "M1"
        )


class OmissionPointsAtTheRightEndpointTests(unittest.TestCase):
    """The two callers omit different keys from different endpoints.

    edge_board's per-model ledger blocks (mark_to_market_by_model,
    quarter_kelly_by_model, growth_*_by_model) are not in /track-record's
    payload, so telling an edge-board consumer to fetch them there sends
    them somewhere the data is not.
    """

    def _payload(self):
        return {"overall": {"n": 1}, "paper_pnl": {"rows": []},
                "mark_to_market_by_model": {"m": []}}

    def test_track_record_points_at_track_record(self):
        out = mcp._summarise_track_record(self._payload())
        self.assertEqual(out["omitted_for_size"]["source"], "/track-record")
        self.assertIn("GET /track-record", out["omitted_for_size"]["detail"])

    def test_edge_board_points_at_edge_board(self):
        out = mcp._summarise_track_record(self._payload(), source="/edge-board")
        self.assertEqual(out["omitted_for_size"]["source"], "/edge-board")
        self.assertIn("GET /edge-board", out["omitted_for_size"]["detail"])
        self.assertNotIn("/track-record", out["omitted_for_size"]["detail"])


class MarketTagsShapeTests(unittest.TestCase):
    """market_tags returned raw Gamma rows: 68% of each one was bookkeeping.

    A real response carried createdAt/updatedAt/requiresTranslation on every
    row plus publishedAt/forceShow/isCarousel/updatedBy on some -- none of it
    describing the tag. Projecting to id/label/slug and taking the full
    100-row page is 36% fewer bytes than the old 50-row payload.
    """

    #: A row of each shape Gamma actually sends, keys and all.
    _ROWS = [
        {
            "id": "537",
            "label": "OpenAI",
            "slug": "openai",
            "publishedAt": "2023-11-17 23:46:12.865+00",
            "updatedBy": 15,
            "createdAt": "2023-11-17T23:46:12.878Z",
            "updatedAt": "2026-04-17T17:23:11.703691Z",
            "requiresTranslation": False,
        },
        {
            "id": "101655",
            "label": "wildfire",
            "slug": "wildfire",
            "forceShow": False,
            "isCarousel": False,
            "createdAt": "2025-01-08T13:49:59.932998Z",
            "updatedAt": "2026-04-17T17:23:11.682475Z",
            "requiresTranslation": False,
        },
        {
            "id": "1512",
            "label": "Caitlin Clark",
            "slug": "caitlin-clark",
            "updatedAt": "2026-04-17T17:23:11.676843Z",
            "requiresTranslation": False,
        },
    ]

    def test_only_the_identifying_keys_survive(self):
        for tag in mcp._summarise_tags(self._ROWS):
            with self.subTest(tag=tag["slug"]):
                self.assertEqual(set(tag), {"id", "label", "slug"})

    def test_no_bookkeeping_key_leaks_through(self):
        seen = {key for tag in mcp._summarise_tags(self._ROWS) for key in tag}
        for key in ("createdAt", "updatedAt", "requiresTranslation",
                    "publishedAt", "forceShow", "isCarousel", "updatedBy"):
            self.assertNotIn(key, seen)

    def test_sorted_by_label_ignoring_case(self):
        """Gamma's order is neither alphabetical nor by activity."""
        labels = [tag["label"] for tag in mcp._summarise_tags(self._ROWS)]
        self.assertEqual(labels, ["Caitlin Clark", "OpenAI", "wildfire"])

    def test_junk_rows_do_not_raise(self):
        self.assertEqual(mcp._summarise_tags(None), [])
        self.assertEqual(mcp._summarise_tags({"tags": []}), [])
        self.assertEqual(mcp._summarise_tags(["text", 7, None]), [])
        self.assertEqual(mcp._summarise_tags([{"label": "x"}]), [{"label": "x"}])

    def test_market_tags_applies_the_projection(self):
        """The wiring, so the projection cannot be dropped without a failure."""
        from analyzing_llm_rationale import market_data

        original = market_data.fetch_polymarket_tags
        market_data.fetch_polymarket_tags = lambda *a, **k: list(self._ROWS)
        try:
            tags = mcp.ForeseaClient().market_tags()
        finally:
            market_data.fetch_polymarket_tags = original
        self.assertEqual([set(tag) for tag in tags], [{"id", "label", "slug"}] * 3)


class TagsPageLimitTests(unittest.TestCase):
    """The page size was whatever Gamma defaulted to, which is not a contract."""

    def test_the_limit_is_sent_explicitly(self):
        from analyzing_llm_rationale import market_data

        sent = {}

        def fake_get_json(url, params=None):
            sent["url"], sent["params"] = url, params
            return []

        original = market_data._get_json
        market_data._get_json = fake_get_json
        try:
            market_data.fetch_polymarket_tags()
        finally:
            market_data._get_json = original
        self.assertEqual(sent["params"], {"limit": 100})

    def test_the_limit_stays_within_what_gamma_honours(self):
        """Gamma caps /tags at 100 and silently truncates anything larger."""
        from analyzing_llm_rationale import market_data

        self.assertLessEqual(market_data.POLYMARKET_TAGS_PAGE_LIMIT, 100)
