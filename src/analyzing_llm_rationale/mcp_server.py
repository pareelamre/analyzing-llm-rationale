from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# tool method (aforecast/...) -> public tool name, for usage logging.
_TOOL_NAMES = {
    "forecast": "foresea_forecast", "aforecast": "foresea_forecast",
    "analyze": "foresea_analyze_market", "aanalyze": "foresea_analyze_market",
    "scan_markets": "foresea_scan_markets", "ascan_markets": "foresea_scan_markets",
    "track_record": "foresea_track_record", "atrack_record": "foresea_track_record",
    "edge_board": "foresea_edge_board", "aedge_board": "foresea_edge_board",
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
        logger.info("mcp_tool_call tool=%s", _tool_name(fn))
        try:
            return fn(*args, **kwargs)
        except ForeseaApiError as exc:
            raise ToolError(f"Foresea API error ({exc.status_code}): {exc.detail}") from exc

    async def _call_tool_async(fn, *args, **kwargs) -> Dict[str, Any]:
        logger.info("mcp_tool_call tool=%s", _tool_name(fn))
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
        """Estimate the probability of any resolvable future question, with a written
        rationale and supporting evidence — and, if you pass a market price or URL,
        the edge vs the market. Use this whenever you need a calibrated probability for
        something that will resolve yes/no (also handles multiple-choice / numeric / date).
        Example: question="Will the Fed cut rates by March 2026?", market_probability=0.4
        → {predicted_answer:"No", confidence:0.62, rationale, evidence_sources,
        market_analysis:{model_probability, edge:+0.14, stance:"model_above_market"}}."""

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
        """End-to-end analysis of one live prediction market. Give a Polymarket slug,
        Kalshi ticker, market URL, or a question — Foresea fetches the current price,
        gathers evidence, forecasts, computes the model-vs-market edge, and returns a
        recommendation. Use when you have a specific market to evaluate (vs scan_markets
        to find them). Example: platform="polymarket", slug="fed-rate-cut-march-2026"
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
        return await _call_tool_async(client.aanalyze, payload)

    @mcp.tool()
    async def foresea_scan_markets(
        platform: str = "polymarket",
        limit: int = 4,
        min_edge: float = 0.1,
        evidence_top_k: int = 3,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover live Polymarket/Kalshi markets Foresea thinks are mispriced: returns
        markets ranked by disagreement (|model − market|), each with the model
        probability, market price, and edge. Use to find forecasting/trading
        opportunities. Example: platform="kalshi", min_edge=0.1
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
    async def foresea_track_record() -> Dict[str, Any]:
        """Foresea's public resolved-forecast track record: accuracy, Brier score,
        calibration (ECE), and skill-vs-market by horizon — the proof of how well its
        forecasts have actually done. Call this to decide how much to trust a Foresea
        forecast before acting on it."""

        return await _call_tool_async(client.atrack_record)

    @mcp.tool()
    async def foresea_edge_board() -> Dict[str, Any]:
        """Live model-vs-market edge board: open markets ranked by disagreement, each
        with the explicit trade (Buy YES/NO @ price, implied odds) and whether
        disagreements that size have historically paid (`skill_significant`). Includes
        by-edge calibration, market lead/lag, and a multi-model paper-trading
        comparison. Use to find — and sanity-check — the strongest current edges."""

        return await _call_tool_async(client.aedge_board)

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
