from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
]

STOOQ_RSS_FEEDS = (
    "https://static.stooq.com/rss/pl/b.rss",
    "https://static.stooq.com/rss/pl/c.rss",
    "https://static.stooq.com/rss/pl/w.rss",
)

# Stooq (Polish stock-market RSS) is not in the defaults — it was query-agnostic
# and leaked irrelevant headlines into evidence. It stays available if explicitly
# configured. `web` is real web search, active when a provider is configured:
# TAVILY_API_KEY, SERPER_API_KEY, BRAVE_API_KEY, or SEARXNG_URL.
DEFAULT_FETCH_SOURCES = ("web", "newsapi", "gdelt", "google-news", "rss")
SOURCE_DIVERSITY_ORDER = ("web", "gdelt", "google-news", "newsapi", "rss", "stooq")
HIGH_CREDIBILITY_SOURCES = {
    "abc news",
    "al jazeera",
    "ap",
    "associated press",
    "axios",
    "bbc",
    "bloomberg",
    "cbs news",
    "cnbc",
    "cnn",
    "dw",
    "financial times",
    "fortune",
    "guardian",
    "npr",
    "politico",
    "reuters",
    "the associated press",
    "the guardian",
    "the new york times",
    "the wall street journal",
    "the washington post",
    "time",
    "usa today",
    "wall street journal",
    "washington post",
}
CHANNEL_CREDIBILITY = {
    "web": 0.74,
    "newsapi": 0.75,
    "gdelt": 0.72,
    "google-news": 0.70,
    "rss": 0.68,
    "stooq": 0.55,
}

# Channels that return generic, query-agnostic headlines (Stooq market RSS,
# publisher homepage RSS) rather than results matched to the question. Their
# articles are only kept when they clear a minimum relevance to the question,
# so they stop padding evidence for unrelated or conversational queries.
_QUERY_AGNOSTIC_CHANNELS = ("stooq", "rss")
_MIN_GENERIC_RELEVANCE = 0.1

# Stooq only carries financial-market data, so it is fetched only when the
# question is finance/markets related.
_FINANCE_HINTS = {
    "stock", "stocks", "share", "shares", "equity", "equities", "index", "indices",
    "nasdaq", "s&p", "sp500", "dow", "ftse", "dax", "nikkei", "wig", "nyse",
    "earnings", "ipo", "dividend", "valuation", "ticker", "revenue", "profit",
    "interest", "rate", "rates", "fed", "ecb", "boe", "inflation", "cpi", "gdp",
    "recession", "currency", "forex", "exchange", "bond", "bonds", "yield",
    "treasury", "oil", "crude", "brent", "wti", "gold", "silver", "commodity",
    "commodities", "crypto", "bitcoin", "btc", "ethereum", "eth", "trading",
    "market", "markets", "bank", "stooq",
}
QUERY_STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "is",
    "it",
    "be",
    "before",
    "after",
    "by",
    "on",
    "or",
    "and",
    "to",
    "of",
    "for",
    "per",
    "share",
    "shares",
    "close",
    "above",
    "below",
    "happen",
    "occur",
    "resolves",
}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _lexical_relevance(query: str, text: str) -> float:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _is_finance_query(text: str) -> bool:
    """True when the question mentions markets/finance, gating Stooq retrieval."""
    terms = set(re.findall(r"[a-z&0-9]+", (text or "").lower()))
    return bool(terms & _FINANCE_HINTS)


def _keyword_search_query(question: str, max_terms: int = 12) -> str:
    terms = re.findall(r"[A-Za-z0-9$]+", question)
    kept = []
    for term in terms:
        normalized = term.lower().strip("$")
        if normalized in QUERY_STOPWORDS:
            continue
        if len(normalized) <= 2 and not normalized.isdigit():
            continue
        kept.append(term.strip("$"))
    return " ".join(kept[:max_terms]) or question


def _source_credibility(article: dict) -> float:
    """Small prior favoring established publishers without excluding other hits."""
    source = (article.get("source") or "").lower()
    channel = article.get("source_channel") or ""
    if source in HIGH_CREDIBILITY_SOURCES:
        return 1.0
    if any(name in source for name in HIGH_CREDIBILITY_SOURCES):
        return 0.9
    return CHANNEL_CREDIBILITY.get(channel, 0.6)


class NewsPipeline:
    """Fetch, summarize, and rank news articles for a forecasting question."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.scads.ai/v1",
        model: str = "openai/gpt-oss-120b",
        embedding_model: str = "all-MiniLM-L6-v2",
        newsapi_key: Optional[str] = None,
        use_query_planner: bool = True,
        fetch_sources: Optional[Sequence[str]] = None,
        summarize_articles: bool = True,
        use_embeddings: bool = True,
    ):
        self._llm = None
        if use_query_planner or summarize_articles:
            from langchain_openai import ChatOpenAI

            resolved_key = api_key or os.environ.get("SCADS_AI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "An API key is required. Set the SCADS_AI_API_KEY environment variable "
                    "or pass api_key= explicitly."
                )
            self._llm = ChatOpenAI(
                model=model,
                api_key=resolved_key,
                base_url=base_url,
                temperature=0.0,
                max_tokens=512,
            )
        self._embedding_model_name = embedding_model
        self._embeddings = None
        self._newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")
        self._brave_key = os.environ.get("BRAVE_API_KEY")
        self._tavily_key = os.environ.get("TAVILY_API_KEY")
        self._serper_key = os.environ.get("SERPER_API_KEY")
        self._searxng_url = os.environ.get("SEARXNG_URL")
        self._use_query_planner = use_query_planner
        self._fetch_sources = tuple(fetch_sources or DEFAULT_FETCH_SOURCES)
        self._summarize_articles = summarize_articles
        self._use_embeddings = use_embeddings

    def _get_embeddings(self):
        if not self._use_embeddings:
            return None
        if self._embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self._embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except (ImportError, Exception):
                pass  # rank() will fall back to original order
        return self._embeddings

    def fetch(self, query: str, top_k: int = 10) -> List[dict]:
        """Return up to top_k raw article dicts from configured news sources."""
        articles: List[dict] = []
        per_source_limit = max(top_k, 10)

        web_configured = any((
            getattr(self, "_tavily_key", None), getattr(self, "_serper_key", None),
            getattr(self, "_brave_key", None), getattr(self, "_searxng_url", None),
        ))
        if "web" in self._fetch_sources and web_configured:
            articles.extend(self._fetch_web(query, limit=per_source_limit))

        if self._newsapi_key and "newsapi" in self._fetch_sources:
            articles.extend(self._fetch_newsapi(query, page_size=per_source_limit))

        if "gdelt" in self._fetch_sources:
            articles.extend(self._fetch_gdelt(query, limit=per_source_limit))

        if "google-news" in self._fetch_sources:
            articles.extend(self._fetch_google_news(query, limit=per_source_limit))

        if "stooq" in self._fetch_sources and _is_finance_query(query):
            articles.extend(self._fetch_stooq(limit=per_source_limit))

        if "rss" in self._fetch_sources:
            articles.extend(self._fetch_rss(limit=per_source_limit))

        seen_urls: set = set()
        unique: List[dict] = []
        for a in articles:
            url = a.get("url", "")
            dedupe_key = url or f"{a.get('source', '')}:{a.get('title', '')}"
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)
            unique.append(a)

        return unique

    def _fetch_web(self, query: str, limit: int = 10) -> List[dict]:
        """General web search. Uses the first configured provider (Tavily,
        Serper, Brave, or a self-hosted SearXNG), else a keyless DuckDuckGo
        fallback. Each provider fails open to an empty list."""
        if getattr(self, "_tavily_key", None):
            return self._web_tavily(query, limit)
        if getattr(self, "_serper_key", None):
            return self._web_serper(query, limit)
        if getattr(self, "_brave_key", None):
            return self._web_brave(query, limit)
        if getattr(self, "_searxng_url", None):
            return self._web_searxng(query, limit)
        return self._web_duckduckgo(query, limit)

    @staticmethod
    def _domain(url: str) -> str:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").replace("www.", "") or "Web"

    def _web_tavily(self, query: str, limit: int = 10) -> List[dict]:
        try:
            import requests
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": self._tavily_key, "query": query,
                      "max_results": min(limit, 20), "search_depth": "basic"},
                timeout=15,
            )
            resp.raise_for_status()
            return [{
                "title": r.get("title") or "", "url": r.get("url") or "",
                "publish_date": r.get("published_date") or "",
                "text": r.get("content") or "", "summary": r.get("content") or r.get("title") or "",
                "source": self._domain(r.get("url") or ""), "source_channel": "web",
            } for r in resp.json().get("results", [])]
        except Exception:
            return []

    def _web_serper(self, query: str, limit: int = 10) -> List[dict]:
        try:
            import requests
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self._serper_key, "Content-Type": "application/json"},
                json={"q": query, "num": min(limit, 20)},
                timeout=15,
            )
            resp.raise_for_status()
            return [{
                "title": r.get("title") or "", "url": r.get("link") or "",
                "publish_date": r.get("date") or "",
                "text": r.get("snippet") or "", "summary": r.get("snippet") or r.get("title") or "",
                "source": self._domain(r.get("link") or ""), "source_channel": "web",
            } for r in resp.json().get("organic", [])]
        except Exception:
            return []

    def _web_brave(self, query: str, limit: int = 10) -> List[dict]:
        try:
            import requests
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(limit, 20)},
                headers={"Accept": "application/json", "X-Subscription-Token": self._brave_key},
                timeout=15,
            )
            resp.raise_for_status()
            return [{
                "title": r.get("title") or "", "url": r.get("url") or "",
                "publish_date": r.get("page_age") or r.get("age") or "",
                "text": r.get("description") or "", "summary": r.get("description") or r.get("title") or "",
                "source": (r.get("profile") or {}).get("name") or self._domain(r.get("url") or ""),
                "source_channel": "web",
            } for r in (resp.json().get("web") or {}).get("results", [])]
        except Exception:
            return []

    def _web_searxng(self, query: str, limit: int = 10) -> List[dict]:
        try:
            import requests
            base = self._searxng_url.rstrip("/")
            resp = requests.get(
                f"{base}/search",
                params={"q": query, "format": "json", "safesearch": 0},
                headers={"User-Agent": "foresea-market-bot/1.0"},
                timeout=15,
            )
            resp.raise_for_status()
            return [{
                "title": r.get("title") or "", "url": r.get("url") or "",
                "publish_date": r.get("publishedDate") or "",
                "text": r.get("content") or "", "summary": r.get("content") or r.get("title") or "",
                "source": self._domain(r.get("url") or ""), "source_channel": "web",
            } for r in resp.json().get("results", [])[:limit]]
        except Exception:
            return []

    def _web_duckduckgo(self, query: str, limit: int = 10) -> List[dict]:
        """Keyless best-effort fallback (HTML endpoint; may be rate-limited)."""
        try:
            from urllib.parse import parse_qs, unquote, urlparse

            import requests
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; foresea/1.0)"},
                timeout=15,
            )
            resp.raise_for_status()
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                return []
            soup = BeautifulSoup(resp.text, "html.parser")
            articles: List[dict] = []
            for res in soup.select(".result"):
                a = res.select_one("a.result__a")
                if not a:
                    continue
                href = a.get("href") or ""
                if "uddg=" in href:  # unwrap DuckDuckGo redirect
                    href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
                title = a.get_text(" ", strip=True)
                if not href or not title:
                    continue
                snippet = res.select_one(".result__snippet")
                text = snippet.get_text(" ", strip=True) if snippet else ""
                articles.append({
                    "title": title, "url": href, "publish_date": "",
                    "text": text, "summary": text or title,
                    "source": self._domain(href), "source_channel": "web",
                })
                if len(articles) >= limit:
                    break
            return articles
        except Exception:
            return []

    def _fetch_newsapi(self, query: str, page_size: int = 10) -> List[dict]:
        try:
            import requests
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                    "apiKey": self._newsapi_key,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "publish_date": item.get("publishedAt") or "",
                    "text": item.get("content") or item.get("description") or "",
                    "summary": item.get("description") or "",
                    "source": item.get("source", {}).get("name") or "",
                    "source_channel": "newsapi",
                })
            return articles
        except Exception:
            return []

    def _fetch_gdelt(self, query: str, limit: int = 20) -> List[dict]:
        try:
            import requests

            resp = requests.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": query,
                    "mode": "ArtList",
                    "format": "json",
                    "maxrecords": min(max(1, limit), 250),
                    "sort": "HybridRel",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title") or "",
                    "url": item.get("url") or "",
                    "publish_date": item.get("seendate") or "",
                    "text": item.get("title") or "",
                    "summary": item.get("title") or "",
                    "source": item.get("source") or item.get("domain") or "GDELT",
                    "source_channel": "gdelt",
                })
            return articles
        except Exception:
            return []

    def _fetch_google_news(self, query: str, limit: int = 20) -> List[dict]:
        params = urlencode({
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        })
        feed_url = f"https://news.google.com/rss/search?{params}"

        try:
            import requests

            resp = requests.get(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 compatible forecasting-evidence-bot/1.0"},
                timeout=15,
            )
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
        except Exception:
            try:
                import feedparser

                feed = feedparser.parse(feed_url)
                entries = feed.entries[:limit]
                return [
                    {
                        "title": entry.get("title") or "",
                        "url": entry.get("link") or "",
                        "publish_date": entry.get("published") or "",
                        "text": entry.get("summary") or "",
                        "summary": entry.get("summary") or "",
                        "source": "Google News",
                        "source_channel": "google-news",
                    }
                    for entry in entries
                ]
            except Exception:
                return []

        articles: List[dict] = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            publish_date = item.findtext("pubDate") or ""
            description = item.findtext("description") or ""
            source = item.findtext("source") or "Google News"
            articles.append({
                "title": title,
                "url": link,
                "publish_date": publish_date,
                "text": description,
                "summary": description,
                "source": source,
                "source_channel": "google-news",
            })
        return articles

    def _fetch_stooq(self, limit: int = 20) -> List[dict]:
        articles: List[dict] = []
        try:
            import requests
        except ImportError:
            return articles

        for feed_url in STOOQ_RSS_FEEDS:
            if len(articles) >= limit:
                break
            try:
                resp = requests.get(
                    feed_url,
                    headers={"User-Agent": "Mozilla/5.0 compatible forecasting-evidence-bot/1.0"},
                    timeout=15,
                )
                resp.raise_for_status()
                root = ElementTree.fromstring(resp.content)
                feed_title = root.findtext("./channel/title") or "Stooq"
                for item in root.findall("./channel/item"):
                    if len(articles) >= limit:
                        break
                    title = item.findtext("title") or ""
                    link = item.findtext("link") or ""
                    description = item.findtext("description") or ""
                    articles.append({
                        "title": title,
                        "url": link,
                        "publish_date": item.findtext("pubDate") or "",
                        "text": description,
                        "summary": description or title,
                        "source": "Stooq",
                        "source_channel": "stooq",
                        "search_query": feed_title,
                    })
            except Exception:
                continue
        return articles

    def _fetch_rss(self, limit: int = 20) -> List[dict]:
        try:
            import feedparser
        except ImportError:
            return []

        articles: List[dict] = []
        for feed_url in RSS_FEEDS:
            if len(articles) >= limit:
                break
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    if len(articles) >= limit:
                        break
                    articles.append({
                        "title": entry.get("title") or "",
                        "url": entry.get("link") or "",
                        "publish_date": entry.get("published") or "",
                        "text": entry.get("summary") or "",
                        "summary": entry.get("summary") or "",
                        "source": feed.feed.get("title") or feed_url,
                        "source_channel": "rss",
                    })
            except Exception:
                continue
        return articles

    def plan_search_query(self, question: str) -> str:
        """Use a small LangChain planner step to turn a forecast into a news query."""
        return self.plan_search_queries(question, max_queries=1)[0]

    def plan_search_queries(self, question: str, max_queries: int = 4) -> List[str]:
        """Generate a compact set of direct and decomposed news queries."""
        max_queries = max(1, max_queries)
        if not self._use_query_planner or self._llm is None:
            return [question]

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You write concise web-news search queries for forecasting questions.",
                ),
                (
                    "user",
                    "Forecasting question:\n{question}\n\n"
                    f"Return up to {max_queries} newline-separated queries, no numbering and no explanation.\n"
                    "Include one direct event query, then decompose the forecast into key drivers, "
                    "base-rate evidence, or resolution-relevant subquestions. "
                    "Preserve important entities, dates, and event terms.",
                ),
            ])
            chain = prompt | self._llm | StrOutputParser()
            planned = chain.invoke({"question": question}).strip()
        except Exception:
            return [question]

        queries: List[str] = []
        for line in planned.splitlines():
            query = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"')
            if not query:
                continue
            if query.lower() in {existing.lower() for existing in queries}:
                continue
            queries.append(query[:200])
            if len(queries) >= max_queries:
                break
        return queries or [question]

    def summarize(self, article: dict) -> str:
        """Summarize a single article using LangChain + SCADS AI LLM."""
        from langchain_core.messages import HumanMessage

        text = (article.get("text") or article.get("summary") or "").strip()
        if not text:
            return article.get("title") or ""

        if len(text) < 200:
            return text

        msg = HumanMessage(
            content=(
                f"Summarize the following news article in 2-3 sentences, "
                f"focusing on the key facts and implications:\n\n{text[:3000]}"
            )
        )
        try:
            response = self._llm.invoke([msg])
            return response.content.strip()
        except Exception:
            return text[:500]

    def rank(self, question: str, articles: List[dict]) -> List[dict]:
        """Return articles sorted by semantic relevance to the question (highest first)."""
        if not articles:
            return []

        embeddings = self._get_embeddings()
        texts = [
            " ".join(
                str(value)
                for value in (a.get("title"), a.get("summary"), a.get("text"))
                if value
            )[:512]
            for a in articles
        ]
        texts = [t if t.strip() else " " for t in texts]

        if embeddings is None:
            relevance_scores = [_lexical_relevance(question, text) for text in texts]
        else:
            try:
                q_vec = np.array(embeddings.embed_query(question))
                doc_vecs = np.array(embeddings.embed_documents(texts))
                relevance_scores = [_cosine_similarity(q_vec, d) for d in doc_vecs]
            except Exception:
                relevance_scores = [_lexical_relevance(question, text) for text in texts]

        scores = []
        for relevance, article in zip(relevance_scores, articles):
            credibility = _source_credibility(article)
            article["source_credibility"] = round(credibility, 2)
            article["relevance"] = round(float(relevance), 4)
            scores.append((0.85 * relevance) + (0.15 * credibility))

        ranked = sorted(
            zip(scores, articles),
            key=lambda x: x[0],
            reverse=True,
        )
        for score, article in ranked:
            article["relevance_score"] = round(score, 4)
        return [a for _, a in ranked]

    def select_diverse_sources(self, ranked: List[dict], top_k: int) -> List[dict]:
        """Pick a relevant final set while avoiding one-source evidence packs."""
        if top_k <= 0 or not ranked:
            return []

        selected: List[dict] = []
        seen_urls: set = set()

        def add(article: dict) -> None:
            url = article.get("url") or f"{article.get('source', '')}:{article.get('title', '')}"
            if url in seen_urls or len(selected) >= top_k:
                return
            # Query-agnostic channels (Stooq, generic RSS) must clear a relevance
            # floor, so they no longer pad evidence for unrelated questions.
            if (
                article.get("source_channel") in _QUERY_AGNOSTIC_CHANNELS
                and article.get("relevance", 1.0) < _MIN_GENERIC_RELEVANCE
            ):
                return
            seen_urls.add(url)
            selected.append(article)

        for channel in SOURCE_DIVERSITY_ORDER:
            if channel not in self._fetch_sources:
                continue
            for article in ranked:
                if article.get("source_channel") == channel:
                    add(article)
                    break

        for article in ranked:
            add(article)

        return selected[:top_k]

    def fetch_summarize_rank(
        self, question: str, top_k: int = 5
    ) -> List[dict]:
        """Full pipeline: fetch → summarize → rank → return top_k."""
        search_queries = self.plan_search_queries(question)
        raw: List[dict] = []
        for search_query in search_queries:
            for article in self.fetch(search_query, top_k=max(top_k, 5)):
                article.setdefault("search_query", search_query)
                raw.append(article)
        if not raw:
            fallback_query = _keyword_search_query(question)
            if fallback_query not in search_queries:
                for article in self.fetch(fallback_query, top_k=top_k * 2):
                    article.setdefault("search_query", fallback_query)
                    raw.append(article)
        for article in raw:
            if self._summarize_articles:
                article["summary"] = self.summarize(article)
            article.setdefault("search_query", search_queries[0])
        ranked = self.rank(question, raw)
        return self.select_diverse_sources(ranked, top_k)
