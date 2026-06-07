"""Discover new outreach targets for Foresea PR-agent.

Combines a static seed list (grows over time) with live discovery from the
GitHub search API (MCP directory / awesome-list repos that accept submissions).

Each discovered target is returned as a dict compatible with OutreachTarget.
Targets already present in the current targets file (matched by endpoint) are
filtered out so only genuinely new ones are returned.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set

# Static seed list — add new directories here.
# transport="webhook" → POST the standard Foresea PR-agent JSON packet.
# transport="json"    → POST a custom `body` dict.
# transport="form"    → POST a `form` dict as multipart/form-data.
STATIC_SEEDS: List[Dict[str, Any]] = [
    # ── MCP-specific registries ───────────────────────────────────────────────
    {
        "name": "Glama",
        "endpoint": "https://glama.ai/api/mcp/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink/mcp/",
            "repositoryUrl": "https://github.com/pareelamre/analyzing-llm-rationale",
            "description": "Remote MCP server for prediction-market forecasting: calibrated YES/NO probabilities, live edge scans, and public track-record verification.",
        },
    },
    {
        "name": "mcp.so",
        "endpoint": "https://mcp.so/api/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink/mcp/",
            "github": "https://github.com/pareelamre/analyzing-llm-rationale",
            "description": "Remote MCP server: calibrated forecasts, live market edge scans, and public track record for prediction markets.",
        },
    },
    {
        "name": "PulseMCP",
        "endpoint": "https://api.pulsemcp.com/v0beta/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "github_url": "https://github.com/pareelamre/analyzing-llm-rationale",
            "contact_email": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "mcp-get",
        "endpoint": "https://mcp-get.com/api/servers/submit",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "foresea",
            "description": "Prediction-market forecasts, edge scans, and public track-record checks via remote MCP.",
            "vendor": "Foresea",
            "sourceUrl": "https://github.com/pareelamre/analyzing-llm-rationale",
            "homepage": "https://foresea.ink",
            "license": "MIT",
        },
    },
    {
        "name": "Smithery registry",
        "endpoint": "https://smithery.ai/api/v1/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "repoUrl": "https://github.com/pareelamre/analyzing-llm-rationale",
            "serverUrl": "https://foresea.ink/mcp/",
        },
    },
    {
        "name": "MCPHunt",
        "endpoint": "https://mcphunt.com/api/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink/mcp/",
            "repo": "https://github.com/pareelamre/analyzing-llm-rationale",
            "description": "Calibrated AI forecasting on prediction markets — forecasts, edge scans, track record.",
        },
    },
    {
        "name": "mcp.run",
        "endpoint": "https://www.mcp.run/api/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink/mcp/",
            "homepage": "https://foresea.ink",
            "description": "Prediction-market forecasts, edge scans, and public track-record checks.",
        },
    },
    {
        "name": "OpenTools.ai",
        "endpoint": "https://opentools.ai/api/mcp/register",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink/mcp/",
            "repo": "https://github.com/pareelamre/analyzing-llm-rationale",
            "description": "Prediction-market intelligence — calibrated forecasts, edge scans, and track-record checks.",
        },
    },
    {
        "name": "Toolbase",
        "endpoint": "https://api.toolbase.ai/v1/mcp/register",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "endpoint": "https://foresea.ink/mcp/",
            "description": "Calibrated AI forecasting on prediction markets.",
            "homepage": "https://foresea.ink",
        },
    },
    # ── AI-agent / LLM tool directories ──────────────────────────────────────
    {
        "name": "AgentNDX",
        "endpoint": "https://agentndx.ai/api/submit",
        "audience": "catalog",
        "transport": "webhook",
    },
    {
        "name": "ACI.dev",
        "endpoint": "https://api.aci.dev/v1/tools/submit",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Calibrated AI forecasting for prediction markets — YES/NO probabilities, edge scans, multi-model track record.",
            "homepage": "https://foresea.ink",
            "mcp_url": "https://foresea.ink/mcp/",
            "github": "https://github.com/pareelamre/analyzing-llm-rationale",
        },
    },
    {
        "name": "Composio MCP",
        "endpoint": "https://backend.composio.dev/api/v1/mcp/servers",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Remote MCP server: calibrated prediction-market forecasts, live edge scans, public track record.",
            "url": "https://foresea.ink/mcp/",
            "homepage": "https://foresea.ink",
        },
    },
    # ── Broader AI tool directories ───────────────────────────────────────────
    {
        "name": "There's An AI For That",
        "endpoint": "https://theresanai.com/api/tools",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "AI forecasting engine for prediction markets. Ask any probability question and get a calibrated YES/NO answer with sources, live market edge, and a public track record.",
            "url": "https://foresea.ink",
            "category": "Research",
            "pricing": "Free",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Futurepedia",
        "endpoint": "https://www.futurepedia.io/api/tool-submissions",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "tagline": "Calibrated AI forecasting for prediction markets",
            "description": "Ask any probability question and get a calibrated YES/NO answer with evidence, live market edge analysis, and a public track record. Includes an MCP server for Claude Desktop and other AI agents.",
            "url": "https://foresea.ink",
            "categories": ["AI Tools", "Finance", "Research"],
            "pricing": "Free",
            "email": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "TopAI.tools",
        "endpoint": "https://topai.tools/api/submit",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink",
            "description": "AI forecasting engine for prediction markets — calibrated probabilities, market edge scans, and public track record.",
            "category": "Research & Analysis",
            "email": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "AI Tools Club",
        "endpoint": "https://aitools.club/api/submit",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "url": "https://foresea.ink",
            "description": "Calibrated AI forecasting for prediction markets with live edge scans and public track record.",
            "tags": ["forecasting", "prediction markets", "MCP", "AI agents"],
        },
    },
    {
        "name": "AIToolsDirectory",
        "endpoint": "https://www.aitoolsdirectory.com/api/submit",
        "audience": "catalog",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "website": "https://foresea.ink",
            "shortDescription": "AI forecasting for prediction markets",
            "description": "Ask any probability question and get a calibrated YES/NO answer with evidence, live market edge, and a public track record. MCP server available for Claude Desktop.",
            "category": "Research",
            "pricing": "Free",
            "email": "pareel.amre@gmail.com",
        },
    },
    # ── Agent orchestration platforms ─────────────────────────────────────────
    {
        "name": "Zapier AI Actions",
        "endpoint": "https://actions.zapier.com/api/v1/tools/register",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea Forecast",
            "description": "Get a calibrated probability for any future event. Pass a question (and optionally a market URL or current market price) to get YES/NO confidence, written rationale, supporting evidence, and model-vs-market edge.",
            "homepage": "https://foresea.ink",
            "mcp_url": "https://foresea.ink/mcp/",
            "openapi_url": "https://foresea.ink/openapi.json",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Relevance AI Tool Store",
        "endpoint": "https://api.relevanceai.com/v1/tools/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Calibrated AI forecasting for prediction markets. Returns YES/NO probabilities, multi-choice or numeric ranges, written rationale, supporting evidence, and live model-vs-market edge. Exposes 5 tools via MCP: foresea_forecast, foresea_analyze_market, foresea_scan_markets, foresea_edge_board, foresea_track_record.",
            "homepage": "https://foresea.ink",
            "mcp_url": "https://foresea.ink/mcp/",
            "api_url": "https://foresea.ink/predict",
            "openapi_url": "https://foresea.ink/openapi.json",
            "category": "Research & Analytics",
            "tags": ["forecasting", "prediction markets", "probability", "finance"],
            "pricing": "Free",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Make (Integromat) App Directory",
        "endpoint": "https://www.make.com/api/v1/apps/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Calibrated probability forecasts for any future event. Use in Make scenarios to forecast outcomes, scan prediction markets for edges, and verify model accuracy against a public track record.",
            "homepage": "https://foresea.ink",
            "api_base": "https://foresea.ink",
            "openapi_url": "https://foresea.ink/openapi.json",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Voiceflow Marketplace",
        "endpoint": "https://api.voiceflow.com/v2/marketplace/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Add calibrated probability forecasting to any Voiceflow agent. Foresea answers questions about the likelihood of future events with evidence-backed YES/NO probabilities, market edge analysis, and a public track record.",
            "homepage": "https://foresea.ink",
            "mcp_url": "https://foresea.ink/mcp/",
            "api_docs": "https://foresea.ink/docs",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Stack AI",
        "endpoint": "https://www.stack-ai.com/api/tools/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Calibrated AI forecasting for prediction markets — YES/NO probabilities with rationale, live market edge scans, and a public track record. Available as a REST API and remote MCP server.",
            "homepage": "https://foresea.ink",
            "api_url": "https://foresea.ink/predict",
            "mcp_url": "https://foresea.ink/mcp/",
            "openapi_url": "https://foresea.ink/openapi.json",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Botpress Hub",
        "endpoint": "https://api.botpress.com/v1/hub/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Give your Botpress agent the ability to forecast future events with calibrated probabilities, scan prediction markets for mispriced opportunities, and cite a public accuracy track record.",
            "homepage": "https://foresea.ink",
            "api_url": "https://foresea.ink/predict",
            "openapi_url": "https://foresea.ink/openapi.json",
            "contact": "pareel.amre@gmail.com",
        },
    },
    {
        "name": "Langflow Store",
        "endpoint": "https://api.langflow.store/v1/components/submit",
        "audience": "agent",
        "transport": "json",
        "body": {
            "name": "Foresea",
            "description": "Langflow component that calls the Foresea prediction-market forecasting API. Returns calibrated probabilities, rationale, evidence sources, and model-vs-market edge for any binary or multi-choice question.",
            "homepage": "https://foresea.ink",
            "api_url": "https://foresea.ink/predict",
            "openapi_url": "https://foresea.ink/openapi.json",
            "github": "https://github.com/pareelamre/analyzing-llm-rationale",
            "contact": "pareel.amre@gmail.com",
        },
    },
]

# GitHub search queries — repos that are likely MCP directories accepting submissions.
_GH_QUERIES = [
    "awesome-mcp-servers topic:mcp",
    "mcp-server-list topic:mcp-server",
    "topic:mcp-server-directory",
    "mcp directory servers site:github.com",
]

_SUBMISSION_PATTERNS = [
    r"https?://[^\s\"']+/submit[^\s\"')>]*",
    r"https?://[^\s\"']+/api/submit[^\s\"')>]*",
    r"https?://[^\s\"']+/add[^\s\"')>]*",
]


def _gh_headers(token: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def discover_from_github(
    token: Optional[str] = None,
    max_repos: int = 20,
    timeout_s: float = 10.0,
) -> List[Dict[str, Any]]:
    """Search GitHub for MCP directory repos; extract submission endpoints from README."""
    try:
        import requests
    except ImportError:
        return []

    found: List[Dict[str, Any]] = []
    seen_endpoints: Set[str] = set()
    session = requests.Session()

    for query in _GH_QUERIES:
        try:
            resp = session.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": max_repos},
                headers=_gh_headers(token),
                timeout=timeout_s,
            )
            if resp.status_code == 403:
                break  # rate limited; stop searching
            if resp.status_code != 200:
                continue
            repos = resp.json().get("items", [])
        except Exception:
            continue

        for repo in repos:
            full_name = repo.get("full_name", "")
            if not full_name:
                continue
            # Fetch README to look for submission endpoints
            try:
                readme_resp = session.get(
                    f"https://raw.githubusercontent.com/{full_name}/main/README.md",
                    headers=_gh_headers(token),
                    timeout=timeout_s,
                )
                if readme_resp.status_code != 200:
                    readme_resp = session.get(
                        f"https://raw.githubusercontent.com/{full_name}/master/README.md",
                        headers=_gh_headers(token),
                        timeout=timeout_s,
                    )
                if readme_resp.status_code != 200:
                    continue
                readme = readme_resp.text
            except Exception:
                continue

            for pattern in _SUBMISSION_PATTERNS:
                for match in re.finditer(pattern, readme, re.IGNORECASE):
                    endpoint = match.group(0).rstrip(".,;)")
                    if endpoint in seen_endpoints:
                        continue
                    # Skip obvious false positives
                    if any(skip in endpoint for skip in ("badge", "shields.io", "img.", "svg", "png")):
                        continue
                    seen_endpoints.add(endpoint)
                    name = repo.get("name", full_name).replace("-", " ").title()
                    found.append({
                        "name": f"{name} (discovered)",
                        "endpoint": endpoint,
                        "audience": "catalog",
                        "transport": "webhook",
                        "_source": f"github:{full_name}",
                    })

        time.sleep(1.0)  # stay inside GitHub's unauthenticated 10 req/min limit

    return found


def discover_all(
    existing_endpoints: Optional[Set[str]] = None,
    github_token: Optional[str] = None,
    include_github: bool = True,
) -> List[Dict[str, Any]]:
    """Return new targets (from seeds + GitHub) not in existing_endpoints."""
    existing = existing_endpoints or set()
    new_targets: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for seed in STATIC_SEEDS:
        ep = seed.get("endpoint", "")
        if not ep or ep in existing or ep in seen:
            continue
        seen.add(ep)
        new_targets.append(seed)

    if include_github:
        for t in discover_from_github(token=github_token):
            ep = t.get("endpoint", "")
            if not ep or ep in existing or ep in seen:
                continue
            seen.add(ep)
            new_targets.append(t)

    return new_targets
