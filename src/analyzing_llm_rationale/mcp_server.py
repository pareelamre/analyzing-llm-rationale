from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _log_mcp_tool_call(tool: str, context: Optional[Dict[str, Any]] = None) -> None:
    """Write a structured JSON line to stdout so Cloud Run indexes it as jsonPayload."""
    record: Dict[str, Any] = {
        "event": "mcp_tool_call",
        "tool": tool,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if context:
        record.update(context)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _mcp_tool_context(fn_name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract a small, loggable summary from tool call arguments."""
    if fn_name == "aforecast":
        payload = args[0] if args else {}
        q = str(payload.get("question", ""))
        return {"question": q[:140] + ("…" if len(q) > 140 else "")}
    if fn_name in ("aanalyze", "aanalyze_resilient"):
        payload = args[0] if args else {}
        return {k: payload[k] for k in ("platform", "slug", "ticker", "market_id", "question") if k in payload}
    if fn_name == "ascan_markets":
        return {k: kwargs[k] for k in ("platform", "query", "min_edge") if k in kwargs}
    if fn_name == "abatch_quotes":
        refs = args[0] if args else kwargs.get("refs") or []
        return {"ref_count": len(refs)}
    if fn_name == "acheck_run":
        return {"client_run_key": args[0] if args else kwargs.get("client_run_key")}
    return {}


# tool method (aforecast/...) -> public tool name, for usage logging.
_TOOL_NAMES = {
    "forecast": "foresea_forecast", "aforecast": "foresea_forecast",
    "analyze": "foresea_analyze_market", "aanalyze": "foresea_analyze_market",
    "aanalyze_resilient": "foresea_analyze_market",
    "scan_markets": "foresea_scan_markets", "ascan_markets": "foresea_scan_markets",
    "track_record": "foresea_track_record", "atrack_record": "foresea_track_record",
    "edge_board": "foresea_edge_board", "aedge_board": "foresea_edge_board",
    "batch_quotes": "foresea_batch_quotes", "abatch_quotes": "foresea_batch_quotes",
    "check_run": "foresea_check_run", "acheck_run": "foresea_check_run",
    "exchange_status": "foresea_exchange_status", "aexchange_status": "foresea_exchange_status",
    "orderbook": "foresea_orderbook", "aorderbook": "foresea_orderbook",
    "market_tags": "foresea_market_tags", "amarket_tags": "foresea_market_tags",
    "price_history": "foresea_price_history", "aprice_history": "foresea_price_history",
    "live_data": "foresea_live_data", "alive_data": "foresea_live_data",
    "polymarket_meta": "foresea_polymarket_meta", "apolymarket_meta": "foresea_polymarket_meta",
    "recent_trades": "foresea_recent_trades", "arecent_trades": "foresea_recent_trades",
    "market_leaderboard": "foresea_market_leaderboard", "amarket_leaderboard": "foresea_market_leaderboard",
}

DEFAULT_FORESEA_BASE_URL = "https://foresea.ink"
DEFAULT_TIMEOUT_S = 120.0
SUPPORTED_TRANSPORTS = ("stdio", "streamable-http", "sse")


class ForeseaApiError(RuntimeError):
    """Raised when the upstream Foresea HTTP API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Foresea API returned {status_code}: {detail}")


def _strip_empty(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if v is not None and v != []}


def _match_run_from_listing(listing: Dict[str, Any], client_run_key: str) -> Dict[str, Any]:
    runs = listing.get("runs") or []
    if not runs:
        raise ForeseaApiError(404, f'No run found for client_run_key="{client_run_key}".')
    return runs[0]


def _timeout_from_env() -> float:
    raw = os.environ.get("FORESEA_MCP_TIMEOUT_S") or os.environ.get("FORESEA_TIMEOUT_S")
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


def build_predict_payload(
    *,
    question: str,
    description: str = "",
    resolution_criteria: str = "",
    question_type: Optional[str] = None,
    options: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    variant: str = "variant0_neutral_baseline",
    attach_evidence: bool = True,
    evidence_top_k: int = 5,
    market_platform: Optional[str] = None,
    market_url: Optional[str] = None,
    market_outcome: Optional[str] = None,
    market_probability: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the JSON body accepted by Foresea's `/predict` endpoint."""

    return _strip_empty({
        "question": question,
        "description": description,
        "resolution_criteria": resolution_criteria,
        "question_type": question_type,
        "options": options or None,
        "categories": categories or None,
        "variant": variant,
        "attach_evidence": attach_evidence,
        "evidence_top_k": evidence_top_k,
        "market_platform": market_platform,
        "market_url": market_url,
        "market_outcome": market_outcome,
        "market_probability": market_probability,
    })


def build_agent_analyze_payload(
    *,
    question: Optional[str] = None,
    platform: Optional[str] = None,
    slug: Optional[str] = None,
    market_id: Optional[str] = None,
    ticker: Optional[str] = None,
    market_probability: Optional[float] = None,
    variant: str = "variant0_neutral_baseline",
    evidence_top_k: int = 5,
    skills: Optional[List[Dict[str, str]]] = None,
    builtin_skills: bool = False,
    ground_in_record: bool = False,
    tool_loop: bool = False,
    max_tool_steps: int = 5,
) -> Dict[str, Any]:
    """Build the JSON body accepted by Foresea's `/agent/analyze` endpoint."""

    return _strip_empty({
        "question": question,
        "platform": platform,
        "slug": slug,
        "market_id": market_id,
        "ticker": ticker,
        "market_probability": market_probability,
        "variant": variant,
        "evidence_top_k": evidence_top_k,
        "skills": skills or None,
        "builtin_skills": builtin_skills,
        "ground_in_record": ground_in_record,
        "tool_loop": tool_loop,
        "max_tool_steps": max_tool_steps,
    })


class ForeseaClient:
    """Small HTTP client used by the MCP tools."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: Optional[float] = None,
        session: Optional[Any] = None,
        async_session: Optional[Any] = None,
    ):
        self.base_url = (base_url or os.environ.get("FORESEA_BASE_URL") or DEFAULT_FORESEA_BASE_URL).rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("FORESEA_API_KEY") or os.environ.get("API_KEY") or "")
        )
        self.timeout_s = timeout_s if timeout_s is not None else _timeout_from_env()
        self._session = session or requests.Session()
        self._async_session = async_session

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "foresea-mcp/0.1",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = self._session.request(
            method,
            self._url(path),
            headers=self._headers(),
            json=json_body,
            params=params,
            timeout=self.timeout_s,
        )
        if response.status_code >= 400:
            raise ForeseaApiError(response.status_code, _response_detail(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise ForeseaApiError(response.status_code, "Foresea returned non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise ForeseaApiError(response.status_code, "Foresea returned a non-object JSON response.")
        return payload

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async def _send(session: Any):
            return await session.request(
                method,
                self._url(path),
                headers=self._headers(),
                json=json_body,
                params=params,
                timeout=self.timeout_s,
            )

        if self._async_session is not None:
            response = await _send(self._async_session)
        else:
            import httpx

            async with httpx.AsyncClient() as session:
                response = await _send(session)
        if response.status_code >= 400:
            raise ForeseaApiError(response.status_code, _response_detail(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise ForeseaApiError(response.status_code, "Foresea returned non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise ForeseaApiError(response.status_code, "Foresea returned a non-object JSON response.")
        return payload

    def forecast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/predict", json_body=payload)

    async def aforecast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._arequest("POST", "/predict", json_body=payload)

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/agent/analyze", json_body=payload)

    async def aanalyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._arequest("POST", "/agent/analyze", json_body=payload)

    async def aanalyze_resilient(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Like aanalyze, but stamps a client_run_key up front and, on a
        client-side timeout, raises an error naming it instead of just failing
        -- the durable run this created may still be progressing server-side
        even though this call gave up waiting. Call check_run/acheck_run with
        the key to pick the result up later instead of restarting."""
        import httpx

        client_run_key = payload.get("client_run_key") or uuid.uuid4().hex
        payload["client_run_key"] = client_run_key
        try:
            return await self.aanalyze(payload)
        except httpx.TimeoutException as exc:
            raise ForeseaApiError(
                504,
                "Timed out waiting for the analysis, but the run may still be in progress "
                f'server-side. Call foresea_check_run(client_run_key="{client_run_key}") in a '
                "moment instead of restarting.",
            ) from exc

    def scan_markets(
        self,
        *,
        platform: str = "polymarket",
        limit: int = 4,
        min_edge: float = 0.1,
        evidence_top_k: int = 3,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = _strip_empty({
            "platform": platform,
            "limit": limit,
            "min_edge": min_edge,
            "evidence_top_k": evidence_top_k,
            "query": query,
        })
        return self._request("GET", "/agent/scan", params=params)

    async def ascan_markets(
        self,
        *,
        platform: str = "polymarket",
        limit: int = 4,
        min_edge: float = 0.1,
        evidence_top_k: int = 3,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = _strip_empty({
            "platform": platform,
            "limit": limit,
            "min_edge": min_edge,
            "evidence_top_k": evidence_top_k,
            "query": query,
        })
        return await self._arequest("GET", "/agent/scan", params=params)

    def batch_quotes(self, refs: List[str]) -> Dict[str, Any]:
        return self._request("GET", "/market/batch", params={"refs": refs})

    async def abatch_quotes(self, refs: List[str]) -> Dict[str, Any]:
        return await self._arequest("GET", "/market/batch", params={"refs": refs})

    def check_run(self, client_run_key: str) -> Dict[str, Any]:
        listing = self._request("GET", "/agent/runs", params={"client_run_key": client_run_key})
        run = _match_run_from_listing(listing, client_run_key)
        if run.get("status") == "running":
            return {"status": "running", "id": run["id"], "detail": "Still in progress -- check back again shortly."}
        return self._request("GET", f"/agent/runs/{run['id']}")

    async def acheck_run(self, client_run_key: str) -> Dict[str, Any]:
        listing = await self._arequest("GET", "/agent/runs", params={"client_run_key": client_run_key})
        run = _match_run_from_listing(listing, client_run_key)
        if run.get("status") == "running":
            return {"status": "running", "id": run["id"], "detail": "Still in progress -- check back again shortly."}
        return await self._arequest("GET", f"/agent/runs/{run['id']}")

    def track_record(self) -> Dict[str, Any]:
        return self._request("GET", "/track-record")

    async def atrack_record(self) -> Dict[str, Any]:
        return await self._arequest("GET", "/track-record")

    def edge_board(self) -> Dict[str, Any]:
        return self._request("GET", "/edge-board")

    async def aedge_board(self) -> Dict[str, Any]:
        return await self._arequest("GET", "/edge-board")

    def openapi(self) -> Dict[str, Any]:
        return self._request("GET", "/openapi.json")

    async def aopenapi(self) -> Dict[str, Any]:
        return await self._arequest("GET", "/openapi.json")

    def exchange_status(self) -> Dict[str, Any]:
        try:
            from analyzing_llm_rationale import market_data
            status = market_data.fetch_kalshi_exchange_status()
            schedule = market_data.fetch_kalshi_exchange_schedule()
            return {"status": status, "schedule": schedule}
        except Exception:
            return {"status": {}, "schedule": {}}

    async def aexchange_status(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.exchange_status)

    def orderbook(self, ticker_or_token: str) -> Dict[str, Any]:
        try:
            from analyzing_llm_rationale import market_data
            ref = ticker_or_token.strip()
            if ref.isdigit() or len(ref) > 20:
                return market_data.fetch_polymarket_orderbook(ref)
            return market_data.fetch_kalshi_orderbook(ref)
        except Exception:
            return {}

    async def aorderbook(self, ticker_or_token: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.orderbook(ticker_or_token))

    def market_tags(self) -> List[Dict[str, Any]]:
        try:
            from analyzing_llm_rationale import market_data
            return market_data.fetch_polymarket_tags()
        except Exception:
            return []

    async def amarket_tags(self) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.market_tags)

    def price_history(self, ticker_or_market: str, series_ticker: str = "") -> List[Dict[str, Any]]:
        try:
            from analyzing_llm_rationale import market_data
            ref = ticker_or_market.strip()
            if "-" in ref and not ref.isdigit():
                return market_data.fetch_kalshi_candlesticks(ref, series_ticker)
            return market_data.fetch_polymarket_price_history(ref)
        except Exception:
            return []

    async def aprice_history(self, ticker_or_market: str, series_ticker: str = "") -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.price_history(ticker_or_market, series_ticker))

    def live_data(self, event_ticker: str = "", data_type: str = "") -> Dict[str, Any]:
        try:
            from analyzing_llm_rationale import market_data
            return market_data.fetch_kalshi_live_data(event_ticker, data_type)
        except Exception:
            return {}

    async def alive_data(self, event_ticker: str = "", data_type: str = "") -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.live_data(event_ticker, data_type))

    def polymarket_meta(self, target: str = "series", market_id: str = "") -> List[Dict[str, Any]]:
        try:
            from analyzing_llm_rationale import market_data
            target_l = target.lower()
            if target_l == "comments":
                return market_data.fetch_polymarket_comments(market_id)
            if target_l in ("sports", "teams"):
                return market_data.fetch_polymarket_sports()
            return market_data.fetch_polymarket_series()
        except Exception:
            return []

    async def apolymarket_meta(self, target: str = "series", market_id: str = "") -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.polymarket_meta(target, market_id))

    def recent_trades(self, platform: str = "kalshi", ticker_or_token: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        try:
            from analyzing_llm_rationale import market_data
            return market_data.fetch_recent_trades(platform, ticker_or_token, limit)
        except Exception:
            return []

    async def arecent_trades(self, platform: str = "kalshi", ticker_or_token: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.recent_trades(platform, ticker_or_token, limit))

    def market_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            from analyzing_llm_rationale import market_data
            return market_data.fetch_trader_leaderboard(limit)
        except Exception:
            return []

    async def amarket_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.market_leaderboard(limit))


def _response_detail(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return (getattr(response, "text", "") or "HTTP error").strip()[:1000]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        return json.dumps(detail if detail is not None else body, sort_keys=True)[:1000]
    return str(body)[:1000]


def _mcp_import_error(exc: ImportError) -> RuntimeError:
    if sys.version_info < (3, 10):
        return RuntimeError(
            "The MCP server requires Python 3.10+ because the official 'mcp' SDK "
            "requires Python >=3.10. Use a Python 3.10/3.11 environment and install "
            "with: pip install -e '.[mcp]'"
        )
    return RuntimeError("The MCP extra is required: pip install -e '.[mcp]'")


def create_mcp_server(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
):
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError as exc:
        raise _mcp_import_error(exc) from exc

    client = ForeseaClient(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    mcp = FastMCP(
        name="Foresea",
        instructions=(
            "Foresea gives calibrated probabilities for any resolvable future event, "
            "with evidence and the edge vs prediction markets — and it publishes a "
            "resolved track record so you can see how right it's been. Use forecast for "
            "any yes/no question, analyze_market to evaluate a specific Polymarket/Kalshi "
            "market, scan_markets / edge_board to find mispriced markets, and "
            "track_record to gauge how much to trust a forecast before acting."
        ),
        website_url=client.base_url,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
        stateless_http=True,
        json_response=True,
    )

    def _tool_name(fn) -> str:
        return _TOOL_NAMES.get(getattr(fn, "__name__", ""), getattr(fn, "__name__", "?"))

    def _call_tool(fn, *args, **kwargs) -> Dict[str, Any]:
        fn_name = getattr(fn, "__name__", "")
        _log_mcp_tool_call(_tool_name(fn), _mcp_tool_context(fn_name, args, kwargs))
        try:
            return fn(*args, **kwargs)
        except ForeseaApiError as exc:
            raise ToolError(f"Foresea API error ({exc.status_code}): {exc.detail}") from exc

    async def _call_tool_async(fn, *args, **kwargs) -> Dict[str, Any]:
        fn_name = getattr(fn, "__name__", "")
        _log_mcp_tool_call(_tool_name(fn), _mcp_tool_context(fn_name, args, kwargs))
        try:
            return await fn(*args, **kwargs)
        except ForeseaApiError as exc:
            raise ToolError(f"Foresea API error ({exc.status_code}): {exc.detail}") from exc

    @mcp.tool()
    async def foresea_forecast(
        question: str,
        description: str = "",
        resolution_criteria: str = "",
        question_type: Optional[str] = None,
        options: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        variant: str = "variant0_neutral_baseline",
        attach_evidence: bool = True,
        evidence_top_k: int = 5,
        market_platform: Optional[str] = None,
        market_url: Optional[str] = None,
        market_outcome: Optional[str] = None,
        market_probability: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call this whenever the user asks about probability, likelihood, or whether
        something will happen. Good triggers: "Will X happen?", "What are the chances
        of Y?", "How likely is Z?", "What's the probability that…", "Do you think X
        will…", "Should I bet on…". Returns a calibrated YES/NO probability (or
        numeric/date range) with written rationale and supporting news evidence.
        If you also have a market price (market_probability) or URL (market_url), pass
        it to get the model-vs-market edge — how mispriced the market is.
        Example: question="Will the Fed cut rates by March 2026?", market_probability=0.4
        → {predicted_answer:"No", confidence:0.62, rationale, evidence_sources,
        market_analysis:{model_probability:0.54, edge:+0.14, stance:"model_above_market"}}
        Handles: binary YES/NO, multiple-choice, numeric ranges, and date questions."""

        payload = build_predict_payload(
            question=question,
            description=description,
            resolution_criteria=resolution_criteria,
            question_type=question_type,
            options=options,
            categories=categories,
            variant=variant,
            attach_evidence=attach_evidence,
            evidence_top_k=evidence_top_k,
            market_platform=market_platform,
            market_url=market_url,
            market_outcome=market_outcome,
            market_probability=market_probability,
        )
        return await _call_tool_async(client.aforecast, payload)

    @mcp.tool()
    async def foresea_analyze_market(
        question: Optional[str] = None,
        platform: Optional[str] = None,
        slug: Optional[str] = None,
        market_id: Optional[str] = None,
        ticker: Optional[str] = None,
        market_probability: Optional[float] = None,
        variant: str = "variant0_neutral_baseline",
        evidence_top_k: int = 5,
        skills: Optional[List[Dict[str, str]]] = None,
        builtin_skills: bool = False,
        ground_in_record: bool = False,
        tool_loop: bool = False,
        max_tool_steps: int = 5,
    ) -> Dict[str, Any]:
        """Call this when the user mentions a specific prediction market by URL, slug,
        or ticker — or asks whether a particular market is over/underpriced. Good
        triggers: "Is this Polymarket fair?", "What's the edge on kalshi:XXXXX?",
        "Should I buy/sell this market?", user pastes a Polymarket or Kalshi URL.
        Fetches the live price, gathers evidence, forecasts, computes model-vs-market
        edge, and returns a recommendation. Use foresea_forecast instead when there is
        no specific live market — just a general probability question.
        Example: platform="polymarket", slug="fed-rate-cut-march-2026"
        → {model_probability, market_probability, edge, stance, recommendation, thesis}."""

        payload = build_agent_analyze_payload(
            question=question,
            platform=platform,
            slug=slug,
            market_id=market_id,
            ticker=ticker,
            market_probability=market_probability,
            variant=variant,
            evidence_top_k=evidence_top_k,
            skills=skills,
            builtin_skills=builtin_skills,
            ground_in_record=ground_in_record,
            tool_loop=tool_loop,
            max_tool_steps=max_tool_steps,
        )
        return await _call_tool_async(client.aanalyze_resilient, payload)

    @mcp.tool()
    async def foresea_scan_markets(
        platform: str = "polymarket",
        limit: int = 4,
        min_edge: float = 0.1,
        evidence_top_k: int = 3,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call this when the user wants to find mispriced or interesting markets, not
        evaluate a specific one. Good triggers: "What should I bet on?", "Find me
        trading opportunities", "Which markets are mispriced right now?", "What's
        Foresea's best edge today?", "Scan Polymarket for opportunities".
        Returns markets ranked by model-vs-market disagreement, each with model
        probability, market price, and edge. For a specific market, use
        foresea_analyze_market instead.
        Example: platform="kalshi", min_edge=0.1
        → [{question, market_probability, model_probability, edge, market_url}]."""

        return await _call_tool_async(
            client.ascan_markets,
            platform=platform,
            limit=limit,
            min_edge=min_edge,
            evidence_top_k=evidence_top_k,
            query=query,
        )

    @mcp.tool()
    async def foresea_batch_quotes(refs: List[str]) -> Dict[str, Any]:
        """Call this when the user wants current price/volume for several markets
        at once -- a watchlist, a portfolio, "check on these 5 markets" -- instead
        of calling foresea_analyze_market once per market. Each ref is
        "platform:ident", e.g. "kalshi:KXFED-25JUN-H" or
        "polymarket:some-market-slug". Every quote carries fetched_at and
        age_seconds so you can judge freshness yourself -- both venues rate-limit
        hard, so don't assume a quote is live without checking age_seconds. One
        bad ref returns an error on that entry only; the rest of the batch still
        succeeds. Up to 50 refs per call.
        Example: refs=["kalshi:KXFED-25JUN-H", "polymarket:fed-cut-2026"]
        → {quotes: [{platform, ident, probability, volume, fetched_at,
        age_seconds, error}], count, truncated}."""

        return await _call_tool_async(client.abatch_quotes, refs)

    @mcp.tool()
    async def foresea_check_run(client_run_key: str) -> Dict[str, Any]:
        """Call this after foresea_analyze_market timed out or errored with a
        message naming a client_run_key -- the research it started may still be
        running server-side. Returns {"status": "running", ...} if it's not done
        yet (call again in a bit), or the full report once it is. Do not call
        this speculatively; only use the client_run_key a prior
        foresea_analyze_market call actually gave you.
        Example: client_run_key="a1b2c3..." → {status:"running", id:"agent_run_..."}
        or the full report once complete."""

        return await _call_tool_async(client.acheck_run, client_run_key)

    @mcp.tool()
    async def foresea_track_record() -> Dict[str, Any]:
        """Call this when the user asks how reliable or accurate Foresea is, or wants
        to know whether to trust a forecast. Good triggers: "How good is Foresea?",
        "What's the track record?", "Has it been right before?", "Is it calibrated?",
        "What's the Brier score?". Returns accuracy, Brier score, calibration (ECE),
        and skill-vs-market broken down by time horizon."""

        return await _call_tool_async(client.atrack_record)

    @mcp.tool()
    async def foresea_edge_board() -> Dict[str, Any]:
        """Call this when the user wants the current top trading opportunities with
        explicit trade directions and historical backing. Good triggers: "What are
        the best bets right now?", "Show me the edge board", "Which model is winning
        the paper-trading competition?", "What's the strongest edge today?",
        "Are these edges statistically significant?". Returns open markets ranked by
        model-vs-market disagreement, each with Buy YES/NO direction, implied odds,
        whether the edge is historically significant, and a multi-model comparison."""

        return await _call_tool_async(client.aedge_board)

    @mcp.tool()
    async def foresea_exchange_status() -> Dict[str, Any]:
        """Call this to check Kalshi exchange operational status (trading active flag)
        and operational hours/schedule."""

        return await _call_tool_async(client.aexchange_status)

    @mcp.tool()
    async def foresea_orderbook(ticker_or_token: str) -> Dict[str, Any]:
        """Call this to fetch the live bids and asks orderbook depth for a Kalshi market
        ticker (e.g. 'KXFED-25JUN-H') or Polymarket YES-token ID."""

        return await _call_tool_async(client.aorderbook, ticker_or_token)

    @mcp.tool()
    async def foresea_market_tags() -> List[Dict[str, Any]]:
        """Call this to list active categories, tags, and classification taxonomy
        across Polymarket prediction markets."""

        return await _call_tool_async(client.amarket_tags)

    @mcp.tool()
    async def foresea_price_history(ticker_or_market: str, series_ticker: str = "") -> List[Dict[str, Any]]:
        """Call this to retrieve historical price series or OHLC candlesticks for a
        market (e.g. Kalshi ticker or Polymarket token/slug)."""

        return await _call_tool_async(client.aprice_history, ticker_or_market, series_ticker)

    @mcp.tool()
    async def foresea_live_data(event_ticker: str = "", data_type: str = "") -> Dict[str, Any]:
        """Call this to fetch real-time sports game statistics, play-by-play data, and
        live event feeds from Kalshi."""

        return await _call_tool_async(client.alive_data, event_ticker, data_type)

    @mcp.tool()
    async def foresea_polymarket_meta(target: str = "series", market_id: str = "") -> List[Dict[str, Any]]:
        """Call this to fetch Polymarket metadata: event series listings, community
        discussion comments for a market, or sports league metadata (target: 'series', 'comments', 'sports')."""

        return await _call_tool_async(client.apolymarket_meta, target, market_id)

    @mcp.tool()
    async def foresea_recent_trades(platform: str = "kalshi", ticker_or_token: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """Call this to fetch recent public executed trades / trade tape (prices, sizes, timestamps) on Kalshi or Polymarket."""

        return await _call_tool_async(client.arecent_trades, platform, ticker_or_token, limit)

    @mcp.tool()
    async def foresea_market_leaderboard(limit: int = 20) -> List[Dict[str, Any]]:
        """Call this to fetch the top profitable prediction market trader leaderboard and rankings from Polymarket."""

        return await _call_tool_async(client.amarket_leaderboard, limit)

    @mcp.resource(
        "foresea://track-record",
        name="Foresea track record",
        mime_type="application/json",
    )
    async def track_record_resource() -> str:
        """Foresea's public resolved-forecast track record."""

        return json.dumps(await _call_tool_async(client.atrack_record), sort_keys=True)

    @mcp.resource(
        "foresea://openapi.json",
        name="Foresea OpenAPI schema",
        mime_type="application/json",
    )
    async def openapi_resource() -> str:
        """Foresea's public OpenAPI schema."""

        return json.dumps(await _call_tool_async(client.aopenapi), sort_keys=True)

    @mcp.prompt()
    def foresea_forecast_prompt(question: str) -> str:
        """Create a compact prompt that asks an agent to use Foresea for a forecast."""

        return (
            "Use the foresea_forecast tool to forecast this resolvable question. "
            "Report the predicted answer, probability or range, rationale, evidence, "
            f"and any model-vs-market edge.\n\nQuestion: {question}"
        )

    @mcp.prompt()
    def foresea_system_prompt() -> str:
        """Returns the recommended system prompt snippet for Claude Desktop users.
        Paste this into your claude_desktop_config.json system prompt to make Claude
        proactively use Foresea tools for forecasting questions."""

        return (
            "You have access to Foresea (foresea.ink), a calibrated AI forecasting "
            "engine. Use foresea_forecast whenever the user asks about probability, "
            "likelihood, chances, or whether something will happen. Use "
            "foresea_analyze_market when the user mentions a specific Polymarket or "
            "Kalshi market. Use foresea_scan_markets when the user wants to find "
            "trading opportunities. Use foresea_edge_board for the current ranked "
            "edge list. Use foresea_track_record to verify Foresea's accuracy before "
            "acting on a forecast. Always call the tool rather than guessing — "
            "Foresea has live evidence and a calibrated track record."
        )

    return mcp


def run_mcp_server(
    *,
    transport: str = "stdio",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: Optional[float] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    streamable_http_path: str = "/mcp",
) -> None:
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(f"transport must be one of {', '.join(SUPPORTED_TRANSPORTS)}")
    mcp = create_mcp_server(
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        host=host,
        port=port,
        streamable_http_path=streamable_http_path,
    )
    mcp.run(transport=transport)
