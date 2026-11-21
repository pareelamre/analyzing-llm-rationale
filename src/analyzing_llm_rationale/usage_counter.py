"""Daily usage counts that survive deploys: pages, API routes, discovery files and MCP tools.

The owner could not see how agents or the API were used: MCP calls only reached Cloud Run's
logs, and the visit history lived on each instance and reset on deploy (2026-11-21). This counts
every request by what it asked for and by the kind of client, in memory, and adds the counts to
Datastore about once a minute in one transaction.

No personal data: counts only. A page load from another site also counts that site's host.
"""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlparse

KIND = "UsageDaily"
FLUSH_EVERY_S = 60.0
MAX_PENDING_KEYS = 400

_CRAWLER = re.compile(
    r"bot\b|bot/|crawl|spider|slurp|gptbot|chatgpt-user|oai-searchbot|claudebot|claude-web|claude-user|"
    r"anthropic-ai|perplexity|bytespider|ccbot|google-extended|googleother|applebot|duckduckbot|yandex|"
    r"amazonbot|meta-externalagent|facebookexternalhit|bingpreview|cohere-ai|diffbot|youbot", re.I)
_SCRIPT = re.compile(
    r"python|httpx|aiohttp|requests|curl|wget|node|axios|undici|go-http|java/|okhttp|openai|anthropic|"
    r"\bmcp\b|claude-code|langchain|llamaindex|postman|insomnia|ruby|php|reqwest|deno|bun/", re.I)

_SKIP_PREFIXES = ("/static/", "/assets/", "/logos/", "/favicon", "/analytics/", "/health", "/ready",
                  "/apple-touch-icon")
_SKIP_SUFFIXES = (".js", ".css", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp", ".woff", ".woff2",
                  ".map", ".txt.gz")
_DISCOVERY = {"/llms.txt", "/llms-full.txt", "/robots.txt", "/sitemap.xml", "/openapi.json", "/agent.json",
              "/docs"}


def client_class(user_agent: Optional[str]) -> str:
    """browser, crawler (search and AI crawlers), agent (scripts, SDKs, MCP clients) or other."""
    ua = (user_agent or "").strip()
    if not ua:
        return "unknown"
    if _CRAWLER.search(ua):
        return "crawler"
    if _SCRIPT.search(ua) and "mozilla" not in ua.lower():
        return "agent"
    if "mozilla" in ua.lower():
        return "browser"
    return "agent" if _SCRIPT.search(ua) else "other"


def _page_name(path: str) -> str:
    if path.startswith("/chat/"):
        return "/chat/*"
    return path or "/"


def classify(method: str, path: str, route_path: Optional[str], content_type: str) -> Optional[Tuple[str, str]]:
    """(group, name) for a request worth counting, or None for assets, probes and analytics."""
    if any(path.startswith(p) for p in _SKIP_PREFIXES) or path.endswith(_SKIP_SUFFIXES):
        return None
    if path in _DISCOVERY or path.startswith("/.well-known/"):
        return "discovery", path
    if path == "/mcp" or path.startswith("/mcp/"):
        return "mcp", method
    if method == "GET" and content_type.startswith("text/html"):
        return "page", _page_name(path)
    return "api", f"{method} {route_path or path}"


def referrer_host(referer: Optional[str], own_hosts: Iterable[str]) -> Optional[str]:
    """The referring site's host for a page load from another site."""
    if not referer:
        return None
    try:
        host = (urlparse(referer).hostname or "").lower()
    except ValueError:
        return None
    if not host or host in set(own_hosts) or host in ("localhost", "127.0.0.1"):
        return None
    return host[4:] if host.startswith("www.") else host


def _day(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc).strftime("%Y-%m-%d")


class UsageCounter:
    """Counts in memory; `maybe_flush` adds them to the store about once a minute."""

    def __init__(self, store: Optional["DatastoreUsageStore"] = None, *, flush_every_s: float = FLUSH_EVERY_S,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.flush_every_s = flush_every_s
        self.clock = clock
        self._pending: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
        self._lock = threading.Lock()
        self._last_flush = clock()
        self._flushing = False

    def add(self, group: str, name: str, client: str = "-", n: int = 1) -> None:
        key = (_day(self.clock()), group, str(name)[:160], client)
        with self._lock:
            self._pending[key] += n

    def due(self) -> bool:
        with self._lock:
            return bool(self._pending) and not self._flushing and (
                self.clock() - self._last_flush >= self.flush_every_s or len(self._pending) >= MAX_PENDING_KEYS)

    def flush(self) -> int:
        """Write pending counts to the store; on failure they are kept for the next flush."""
        with self._lock:
            if self._flushing or not self._pending:
                return 0
            batch, self._pending = dict(self._pending), defaultdict(int)
            self._flushing = True
        try:
            if self.store is not None:
                self.store.add(batch)
            else:  # no durable store (local runs, tests): keep counting in memory
                with self._lock:
                    for key, n in batch.items():
                        self._pending[key] += n
            return len(batch)
        except Exception:
            with self._lock:
                for key, n in batch.items():
                    self._pending[key] += n
            raise
        finally:
            with self._lock:
                self._flushing = False
                self._last_flush = self.clock()

    def pending(self) -> Dict[Tuple[str, str, str, str], int]:
        with self._lock:
            return dict(self._pending)

    def summary(self, days: int = 30) -> Dict[str, Any]:
        stored = self.store.load(days) if self.store is not None else {}
        merged: Dict[Tuple[str, str, str, str], int] = defaultdict(int, stored)
        cutoff = (date.fromisoformat(_day(self.clock())) - timedelta(days=days - 1)).isoformat()
        for key, n in self.pending().items():
            if key[0] >= cutoff:
                merged[key] += n
        return summarize(merged, days=days)


def summarize(counts: Mapping[Tuple[str, str, str, str], int], *, days: int) -> Dict[str, Any]:
    by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tops: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    clients: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (day, group, name, client), n in counts.items():
        by_day[day][group] += n
        tops[group][name] += n
        if group not in ("referrer", "mcp_tool"):
            clients[group][client] += n

    def top(group: str, limit: int = 25) -> list:
        return [{"name": k, "count": v} for k, v in sorted(tops[group].items(), key=lambda kv: -kv[1])[:limit]]

    discovery_by_client: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_day_key, group, name, client), n in counts.items():
        if group == "discovery":
            discovery_by_client[name][client] += n
    return {
        "days": days,
        "first_day": min(by_day) if by_day else None,
        "by_day": [{"day": d, **dict(g)} for d, g in sorted(by_day.items(), reverse=True)],
        "totals": {g: sum(v.values()) for g, v in tops.items()},
        "clients": {g: dict(v) for g, v in clients.items()},
        "pages": top("page"),
        "api": top("api"),
        "discovery": {name: dict(v) for name, v in sorted(discovery_by_client.items())},
        "mcp_tools": top("mcp_tool", 50),
        "referrers": top("referrer"),
        "note": ("Counts only, no personal data. Clients are classed by user agent: browser, crawler "
                 "(search and AI crawlers), agent (scripts, SDKs and MCP clients), other."),
    }


class DatastoreUsageStore:
    """UsageDaily entities keyed day|group|name|client, incremented in one transaction."""

    def __init__(self, client: Any):
        self.client = client

    @staticmethod
    def _name(key: Tuple[str, str, str, str]) -> str:
        return "|".join(key)

    def add(self, batch: Mapping[Tuple[str, str, str, str], int]) -> None:
        from analyzing_llm_rationale import datastore_backend as _ds

        items = list(batch.items())
        for i in range(0, len(items), 200):
            chunk = items[i:i + 200]
            keys = [self.client.key(KIND, self._name(k)) for k, _ in chunk]
            with self.client.transaction():
                existing = {e.key.name if hasattr(e.key, "name") else None: e
                            for e in self.client.get_multi(keys) if e is not None}
                out = []
                for (key, n), dkey in zip(chunk, keys):
                    entity = existing.get(self._name(key))
                    if entity is None:
                        entity = _ds.Entity(dkey, exclude_from_indexes=("name", "client", "count"))
                        day, group, name, client = key
                        entity.update(day=day, group=group, name=name, client=client, count=0)
                    entity["count"] = int(entity.get("count", 0)) + int(n)
                    out.append(entity)
                self.client.put_multi(out)

    def load(self, days: int) -> Dict[Tuple[str, str, str, str], int]:
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days - 1)).isoformat()
        query = self.client.query(kind=KIND)
        query.add_filter("day", ">=", cutoff)
        out: Dict[Tuple[str, str, str, str], int] = {}
        for e in query.fetch():
            out[(e.get("day"), e.get("group"), e.get("name"), e.get("client"))] = int(e.get("count", 0))
        return out
