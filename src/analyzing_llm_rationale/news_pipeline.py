from __future__ import annotations

import concurrent.futures
import os
import re
import threading
from typing import Callable, List, Optional, Sequence
from urllib.parse import urlencode, urlparse, urlunparse
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
# configured. `web` prefers a configured provider (Tavily, Serper, Brave, or
# SearXNG) and otherwise uses the existing keyless DuckDuckGo fallback.
DEFAULT_FETCH_SOURCES = ("web", "newsapi", "gdelt", "google-news", "rss", "open-meteo")
SOURCE_DIVERSITY_ORDER = ("web", "gdelt", "google-news", "newsapi", "rss", "stooq", "fred", "open-meteo")
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
# Social-media/UGC domains are excluded from web-search evidence entirely --
# unlike news outlets, a single post or thread carries no editorial vetting,
# and the model's "always name the source domain" rule (prompts/system.txt)
# would otherwise cite them as if they were a normal news source.
BLOCKED_WEB_DOMAINS = frozenset({
    "reddit.com",
    "redd.it",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "quora.com",
    "pinterest.com",
})

CHANNEL_CREDIBILITY = {
    "web": 0.74,
    "newsapi": 0.75,
    "gdelt": 0.72,
    "google-news": 0.70,
    "rss": 0.68,
    "stooq": 0.55,
    "fred": 0.90,       # St. Louis Fed official data
    "open-meteo": 0.88, # ECMWF/GFS ensemble forecast
}

# Channels that return generic, query-agnostic headlines (Stooq market RSS,
# publisher homepage RSS) rather than results matched to the question. Their
# articles are only kept when they clear a minimum relevance to the question,
# so they stop padding evidence for unrelated or conversational queries.
_QUERY_AGNOSTIC_CHANNELS = ("stooq", "rss")
_MIN_GENERIC_RELEVANCE = 0.1

# Open-Meteo is fetched only for weather-related questions.
_WEATHER_HINTS = {
    "snow", "snowing", "snowfall", "blizzard", "snowstorm",
    "rain", "rainfall", "rainy", "raining", "drizzle",
    "temperature", "temp", "degrees", "fahrenheit", "celsius",
    "weather", "forecast", "precipitation",
    "hurricane", "tornado", "typhoon", "cyclone", "tropical",
    "flood", "flooding", "storm", "thunderstorm", "lightning",
    "frost", "freeze", "freezing", "ice", "icy", "hail",
    "wind", "windspeed", "gust",
    "drought", "humid", "humidity", "heatwave",
}

# Pattern to extract a location from weather questions like
# "Will it snow in Chicago on Dec 25?" → "Chicago"
_WEATHER_LOC_RE = re.compile(
    r'\b(?:in|at|for|near|around)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})'
)

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


# Function words carry no topical signal; counting them lets irrelevant junk
# (a meme video, an unrelated PDF) score nonzero just for containing "the".
_RELEVANCE_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "at",
    "as", "is", "are", "be", "was", "were", "will", "would", "can", "could",
    "this", "that", "these", "those", "it", "its", "with", "from", "into",
    "over", "under", "up", "down", "out", "than", "then", "before", "after",
    "do", "does", "did", "have", "has", "had", "not", "no", "yes", "if", "any",
    "some", "more", "most", "who", "what", "when", "where", "which", "how",
})


def _lexical_relevance(query: str, text: str) -> float:
    """Fraction of the query's *content* words that appear in the text."""
    query_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower())
                   if t not in _RELEVANCE_STOPWORDS}
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _content_terms(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _RELEVANCE_STOPWORDS]


def _bm25_scores(query_terms: List[str], doc_token_lists: List[List[str]],
                 k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Classic BM25 over a small candidate set — the lexical half of hybrid
    retrieval (catches exact keyword/entity matches dense embeddings can miss)."""
    import math

    n = len(doc_token_lists)
    if n == 0 or not query_terms:
        return [0.0] * n
    df: dict = {}
    for tokens in doc_token_lists:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    avgdl = sum(len(t) for t in doc_token_lists) / n or 1.0
    scores: List[float] = []
    for tokens in doc_token_lists:
        tf: dict = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        dl = len(tokens) or 1
        s = 0.0
        for q in query_terms:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
            s += idf * (tf[q] * (k1 + 1)) / (tf[q] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _rrf_fuse(orderings: List[List[int]], n: int, k: int = 60) -> List[float]:
    """Reciprocal Rank Fusion: combine several rankings (each a list of doc
    indices, best-first) into one score per doc. The standard way to fuse
    lexical + dense retrieval."""
    score = [0.0] * n
    for order in orderings:
        for rank, idx in enumerate(order):
            score[idx] += 1.0 / (k + rank + 1)
    return score


def _is_finance_query(text: str) -> bool:
    """True when the question mentions markets/finance, gating Stooq retrieval."""
    terms = set(re.findall(r"[a-z&0-9]+", (text or "").lower()))
    return bool(terms & _FINANCE_HINTS)


def _is_weather_query(text: str) -> bool:
    """True when the question is about weather conditions, gating Open-Meteo retrieval."""
    terms = set(re.findall(r"[a-z]+", (text or "").lower()))
    return bool(terms & _WEATHER_HINTS)


def _extract_weather_location(query: str) -> str:
    """Extract a city/place name from a weather question.

    Tries preposition-anchored extraction first ("in Chicago"), then falls
    back to the first capitalized non-question-word in the text.
    """
    m = _WEATHER_LOC_RE.search(query)
    if m:
        return m.group(1)
    _QW = {"Will", "When", "What", "Which", "Is", "Are", "Does", "Can", "Would", "Should"}
    for word in query.split():
        clean = word.strip("?,.'\"")
        if clean and clean[0].isupper() and clean not in _QW and len(clean) > 2:
            return clean
    return ""


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


def _normalize_article_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return url.strip()
    path = parsed.path.rstrip("/") or parsed.path
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower().removeprefix("www."),
        path,
        "",
        parsed.query,
        "",
    ))


def _normalize_article_title(title: str) -> str:
    title = re.sub(r"\s+", " ", (title or "").lower()).strip()
    title = re.sub(r"\s[-|–—:]\s[^-|–—:]{2,80}$", "", title).strip()
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def _article_dedupe_keys(article: dict) -> List[str]:
    keys: List[str] = []
    url_key = _normalize_article_url(str(article.get("url") or ""))
    if url_key:
        keys.append(f"url:{url_key}")
    title_key = _normalize_article_title(str(article.get("title") or ""))
    if title_key:
        keys.append(f"title:{title_key}")
    if not keys:
        keys.append(f"source-title:{article.get('source', '')}:{article.get('title', '')}")
    return keys


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
        min_relevance: float = 0.0,
        embed_fn: "Optional[Callable[[List[str]], List[List[float]]]]" = None,
        rerank_fn: "Optional[Callable[[str, List[str]], List[float]]]" = None,
        rerank_top_k: int = 12,
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
        self._fred_key = os.environ.get("FRED_API_KEY")
        self._use_query_planner = use_query_planner
        self._fetch_sources = tuple(fetch_sources or DEFAULT_FETCH_SOURCES)
        self._summarize_articles = summarize_articles
        self._use_embeddings = use_embeddings
        # Optional shared embedder (text list -> vectors). When provided, ranking
        # is SEMANTIC and reuses one model instance (e.g. the server's mounted
        # embedder) instead of loading a second copy. Falls back to lexical.
        self._embed_fn = embed_fn
        # Optional cross-encoder reranker (query, texts) -> per-text relevance
        # scores. When set, the top candidates are reranked for precision.
        self._rerank_fn = rerank_fn
        self._rerank_top_k = max(1, int(rerank_top_k))
        # Drop any source below this topical-relevance floor before it can be
        # used as evidence or cited — so irrelevant junk can't ground/hallucinate
        # the forecast. If nothing clears the floor, no sources are returned.
        self._min_relevance = max(0.0, float(min_relevance))

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

    def _embed_texts(self, texts: List[str]):
        """Embed texts to np vectors for semantic ranking, or None to signal a
        lexical fallback. Prefers an injected shared embedder; else a local
        sentence-transformers model when ``use_embeddings`` is on."""
        embed_fn = getattr(self, "_embed_fn", None)
        if embed_fn is not None:
            try:
                vecs = embed_fn(list(texts))
            except Exception:
                vecs = None
            if vecs and all(len(v) for v in vecs):
                return np.array(vecs, dtype=float)
            return None
        embeddings = self._get_embeddings()
        if embeddings is None:
            return None
        try:
            return np.array(embeddings.embed_documents(list(texts)), dtype=float)
        except Exception:
            return None

    def fetch(self, query: str, top_k: int = 10) -> List[dict]:
        """Return up to top_k raw article dicts from configured news sources."""
        articles: List[dict] = []
        per_source_limit = max(top_k, 10)

        if "web" in self._fetch_sources:
            articles.extend(self._fetch_web(query, limit=per_source_limit))

        if self._newsapi_key and "newsapi" in self._fetch_sources:
            articles.extend(self._fetch_newsapi(query, page_size=per_source_limit))

        if "gdelt" in self._fetch_sources:
            articles.extend(self._fetch_gdelt(query, limit=per_source_limit))

        if "google-news" in self._fetch_sources:
            articles.extend(self._fetch_google_news(query, limit=per_source_limit))

        if "stooq" in self._fetch_sources and _is_finance_query(query):
            articles.extend(self._fetch_stooq(limit=per_source_limit))

        if "fred" in self._fetch_sources and self._fred_key and _is_finance_query(query):
            articles.extend(self._fetch_fred(query, limit=5))

        if "open-meteo" in self._fetch_sources and _is_weather_query(query):
            articles.extend(self._fetch_open_meteo(query))

        if "rss" in self._fetch_sources:
            articles.extend(self._fetch_rss(limit=per_source_limit))

        seen_keys: set = set()
        unique: List[dict] = []
        for a in articles:
            dedupe_keys = _article_dedupe_keys(a)
            if any(key in seen_keys for key in dedupe_keys):
                continue
            seen_keys.update(dedupe_keys)
            unique.append(a)

        return unique

    def _fetch_web(self, query: str, limit: int = 10) -> List[dict]:
        """General web search. Uses the first configured provider — preferring a
        self-hosted SearXNG, then Tavily, Serper, Brave — else a keyless
        DuckDuckGo fallback. Each provider fails open to an empty list.
        Social-media/UGC results are filtered out (see BLOCKED_WEB_DOMAINS) --
        a provider whose results are entirely blocked domains is treated as
        having returned nothing, so the next provider in the chain is tried."""
        providers = (
            ("_searxng_url", self._web_searxng),
            ("_tavily_key", self._web_tavily),
            ("_serper_key", self._web_serper),
            ("_brave_key", self._web_brave),
        )
        for setting, provider in providers:
            if getattr(self, setting, None):
                articles = self._filter_blocked_domains(provider(query, limit))
                if articles:
                    return articles
        articles = self._filter_blocked_domains(self._web_duckduckgo(query, limit))
        if articles:
            return articles
        return self._filter_blocked_domains(self._web_ap_news(query, limit))

    @staticmethod
    def _domain(url: str) -> str:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").replace("www.", "") or "Web"

    @staticmethod
    def _filter_blocked_domains(articles: List[dict]) -> List[dict]:
        def _blocked(source: str) -> bool:
            domain = (source or "").lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain in BLOCKED_WEB_DOMAINS or any(
                domain.endswith("." + d) for d in BLOCKED_WEB_DOMAINS
            )
        return [a for a in articles if not _blocked(a.get("source", ""))]

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

    def _parse_duckduckgo_results(self, html: str, limit: int) -> List[dict]:
        """Parse both the HTML and Lite result layouts."""
        try:
            from urllib.parse import parse_qs, unquote, urlparse

            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup = BeautifulSoup(html, "html.parser")
        articles: List[dict] = []
        seen_urls: set[str] = set()
        for anchor in soup.select("a.result__a, a.result-link"):
            href = anchor.get("href") or ""
            if href.startswith("//"):
                href = f"https:{href}"
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            parsed = urlparse(href)
            title = anchor.get_text(" ", strip=True)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or "duckduckgo.com" in parsed.netloc
                or not title
                or href in seen_urls
            ):
                continue

            snippet = None
            result = anchor.find_parent(class_="result")
            if result is not None:
                snippet = result.select_one(".result__snippet")
            else:
                row = anchor.find_parent("tr")
                if row is not None:
                    snippet_row = row.find_next_sibling("tr")
                    if snippet_row is not None:
                        snippet = snippet_row.select_one(".result-snippet")
            text = snippet.get_text(" ", strip=True) if snippet else ""
            articles.append({
                "title": title, "url": href, "publish_date": "",
                "text": text, "summary": text or title,
                "source": self._domain(href), "source_channel": "web",
            })
            seen_urls.add(href)
            if len(articles) >= limit:
                break
        return articles

    def _web_duckduckgo(self, query: str, limit: int = 10) -> List[dict]:
        """Keyless fallback with independent HTML and Lite endpoints."""
        try:
            import requests
        except ImportError:
            return []

        endpoints = (
            "https://html.duckduckgo.com/html/",
            "https://lite.duckduckgo.com/lite/",
        )
        for endpoint in endpoints:
            try:
                resp = requests.get(
                    endpoint,
                    params={"q": query},
                    headers={"User-Agent": "Foresea/1.0"},
                    timeout=15,
                    allow_redirects=True,
                )
                resp.raise_for_status()
                articles = self._parse_duckduckgo_results(resp.text, limit)
                if articles:
                    return articles
            except Exception:
                continue
        return []

    def _web_ap_news(self, query: str, limit: int = 10) -> List[dict]:
        """Independent publisher-search fallback when search engines throttle."""
        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(
                "https://apnews.com/search",
                params={"q": query},
                headers={"User-Agent": "Foresea/1.0"},
                timeout=15,
                allow_redirects=True,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return []

        articles: List[dict] = []
        result_limit = min(limit, 3)
        seen_urls: set[str] = set()
        for item in soup.select(".PageList-items-item"):
            anchor = item.select_one(".PagePromo-title a[href]")
            if anchor is None:
                continue
            title = anchor.get_text(" ", strip=True)
            href = anchor.get("href") or ""
            if not title or not href.startswith(("http://", "https://")) or href in seen_urls:
                continue
            description = item.select_one(".PagePromo-description")
            text = description.get_text(" ", strip=True) if description else ""
            relevance = _lexical_relevance(query, f"{title} {text}")
            if relevance < max(0.35, getattr(self, "_min_relevance", 0.0)):
                continue
            articles.append({
                "title": title, "url": href, "publish_date": "",
                "text": text, "summary": text or title,
                "source": "Associated Press", "source_channel": "web",
            })
            seen_urls.add(href)
            if len(articles) >= result_limit:
                break
        return articles

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

    def _fetch_fred(self, query: str, limit: int = 5) -> List[dict]:
        """Fetch FRED economic time series matching the query as evidence items.

        Searches the St. Louis Fed API for series relevant to the question, then
        fetches recent observations for the top matches. Only called when
        _is_finance_query() is True and FRED_API_KEY is set.
        """
        try:
            import requests
        except ImportError:
            return []

        _FRED_BASE = "https://api.stlouisfed.org/fred"
        articles: List[dict] = []

        try:
            search_text = _keyword_search_query(query, max_terms=8)
            resp = requests.get(
                f"{_FRED_BASE}/series/search",
                params={
                    "search_text": search_text,
                    "api_key": self._fred_key,
                    "file_type": "json",
                    "limit": limit,
                    "order_by": "popularity",
                    "sort_order": "desc",
                },
                timeout=15,
            )
            resp.raise_for_status()
            series_list = resp.json().get("seriess", [])
        except Exception:
            return []

        for series in series_list[:limit]:
            series_id = series.get("id", "")
            if not series_id:
                continue
            title = series.get("title", series_id)
            units = series.get("units_short") or series.get("units") or ""
            frequency = series.get("frequency_short") or series.get("frequency") or ""

            try:
                obs_resp = requests.get(
                    f"{_FRED_BASE}/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": self._fred_key,
                        "file_type": "json",
                        "limit": 12,
                        "sort_order": "desc",
                    },
                    timeout=15,
                )
                obs_resp.raise_for_status()
                observations = obs_resp.json().get("observations", [])
            except Exception:
                continue

            obs_lines = [
                f"{o['date']}: {o['value']} {units}".strip()
                for o in observations[:8]
                if o.get("value") and o["value"] != "."
            ]
            if not obs_lines:
                continue

            freq_label = f" ({frequency})" if frequency else ""
            text = f"{title}{freq_label}\nRecent values (most recent first):\n" + "\n".join(obs_lines)
            articles.append({
                "title": f"FRED: {title}",
                "url": f"https://fred.stlouisfed.org/series/{series_id}",
                "text": text,
                "summary": obs_lines[0],
                "source": "FRED (St. Louis Fed)",
                "source_channel": "fred",
                "publish_date": observations[0].get("date", "") if observations else "",
            })

        return articles

    def _fetch_open_meteo(self, query: str) -> List[dict]:
        """Fetch a 14-day weather forecast from Open-Meteo for the location in query.

        Geocodes the city extracted from the question, fetches temperature,
        precipitation probability, snowfall, rain, and wind. Returns one
        evidence item with the full day-by-day table so the LLM can reason
        against the specific date/threshold in the question.
        Only called when _is_weather_query() is True.
        """
        try:
            import requests
        except ImportError:
            return []

        location = _extract_weather_location(query)
        if not location:
            return []

        try:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            geo.raise_for_status()
            results = geo.json().get("results", [])
            if not results:
                return []
            r = results[0]
            lat, lon = r["latitude"], r["longitude"]
            loc_name = ", ".join(filter(None, [
                r.get("name", location),
                r.get("admin1", ""),
                r.get("country", ""),
            ]))
        except Exception:
            return []

        try:
            fc = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ",".join([
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "precipitation_sum",
                        "snowfall_sum",
                        "rain_sum",
                        "windspeed_10m_max",
                    ]),
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                    "precipitation_unit": "inch",
                    "forecast_days": 14,
                    "timezone": "auto",
                },
                timeout=10,
            )
            fc.raise_for_status()
            daily = fc.json().get("daily", {})
        except Exception:
            return []

        dates = daily.get("time", [])
        if not dates:
            return []

        def _g(key: str, i: int):
            col = daily.get(key, [])
            return col[i] if i < len(col) else None

        lines = []
        for i, date in enumerate(dates):
            tmax, tmin = _g("temperature_2m_max", i), _g("temperature_2m_min", i)
            pprob = _g("precipitation_probability_max", i)
            snow = _g("snowfall_sum", i)
            rain = _g("rain_sum", i)
            wind = _g("windspeed_10m_max", i)
            parts = [date]
            if tmax is not None and tmin is not None:
                parts.append(f"temp {tmin:.0f}–{tmax:.0f}°F")
            if pprob is not None:
                parts.append(f"precip {pprob:.0f}%")
            if snow and snow > 0:
                parts.append(f"snow {snow:.1f}in")
            if rain and rain > 0:
                parts.append(f"rain {rain:.2f}in")
            if wind is not None:
                parts.append(f"wind {wind:.0f}mph")
            lines.append("  ".join(parts))

        text = f"Open-Meteo 14-day forecast for {loc_name}:\n" + "\n".join(lines)
        return [{
            "title": f"Weather forecast: {loc_name} (14-day)",
            "url": "https://open-meteo.com/",
            "text": text,
            "summary": lines[0] if lines else "",
            "source": "Open-Meteo",
            "source_channel": "open-meteo",
            "publish_date": dates[0] if dates else "",
        }]

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
        """Rank articles by relevance using **hybrid retrieval** (dense cosine +
        BM25, fused with Reciprocal Rank Fusion) and, when a cross-encoder
        ``rerank_fn`` is configured, a **reranking** precision stage over the top
        candidates. ``article['relevance']`` holds the semantic (cosine) score,
        which drives the downstream relevance floor; ordering uses the hybrid /
        reranked score."""
        if not articles:
            return []

        texts = [
            " ".join(
                str(value)
                for value in (a.get("title"), a.get("summary"), a.get("text"))
                if value
            )[:512]
            for a in articles
        ]
        texts = [t if t.strip() else " " for t in texts]
        n = len(texts)

        # Dense semantic relevance (also the junk-filter signal).
        vectors = self._embed_texts([question] + texts)
        if vectors is None:
            dense_rel = [_lexical_relevance(question, t) for t in texts]
            has_dense = False
        else:
            q_vec, doc_vecs = vectors[0], vectors[1:]
            dense_rel = [max(0.0, min(1.0, float(_cosine_similarity(q_vec, d)))) for d in doc_vecs]
            has_dense = True

        # Keep a lexical relevance signal even when dense embeddings are available.
        # In production the evidence floor is used to suppress junk sources, but a
        # weak or miscalibrated embedding model can otherwise zero out every
        # article despite strong exact keyword/entity matches.
        lexical_rel = [_lexical_relevance(question, t) for t in texts]

        # Lexical BM25 (the other half of hybrid retrieval).
        q_terms = _content_terms(question)
        bm25 = _bm25_scores(q_terms, [re.findall(r"[a-z0-9]+", t.lower()) for t in texts])

        # Fuse dense + BM25 rankings via RRF (or fall back to whichever exists).
        dense_order = sorted(range(n), key=lambda i: dense_rel[i], reverse=True)
        if has_dense and any(bm25):
            bm25_order = sorted(range(n), key=lambda i: bm25[i], reverse=True)
            order_score = _rrf_fuse([dense_order, bm25_order], n)
        elif has_dense:
            order_score = list(dense_rel)
        else:
            order_score = bm25 if any(bm25) else list(dense_rel)
        order = sorted(range(n), key=lambda i: order_score[i], reverse=True)

        # Cross-encoder reranking of the top candidates (precision stage).
        rerank_fn = getattr(self, "_rerank_fn", None)
        rerank_top_k = getattr(self, "_rerank_top_k", 12)
        rerank_scores: dict = {}
        if rerank_fn is not None and len(order) > 1:
            head = order[:rerank_top_k]
            try:
                rr = rerank_fn(question, [texts[i] for i in head])
            except Exception:
                rr = None
            if rr and len(rr) == len(head):
                rerank_scores = {i: float(s) for i, s in zip(head, rr)}
                head = [i for _, i in sorted(zip(rr, head), key=lambda x: x[0], reverse=True)]
                order = head + order[rerank_top_k:]

        for i, article in enumerate(articles):
            credibility = _source_credibility(article)
            article["source_credibility"] = round(credibility, 2)
            article["semantic_relevance"] = round(dense_rel[i], 4)
            article["lexical_relevance"] = round(lexical_rel[i], 4)
            # Relevance floor should fail open to exact lexical matches rather than
            # dropping all evidence when the embedding signal is weak.
            article["relevance"] = round(max(dense_rel[i], lexical_rel[i]), 4)
            article["bm25"] = round(bm25[i], 4)
            if i in rerank_scores:
                article["rerank_score"] = round(rerank_scores[i], 4)
            article["relevance_score"] = round(max(0.0, min(1.0, 0.85 * dense_rel[i] + 0.15 * credibility)), 4)
        return [articles[i] for i in order]

    def select_diverse_sources(self, ranked: List[dict], top_k: int) -> List[dict]:
        """Pick a relevant final set while avoiding one-source evidence packs."""
        if top_k <= 0 or not ranked:
            return []

        selected: List[dict] = []
        seen_keys: set = set()

        def add(article: dict) -> bool:
            dedupe_keys = _article_dedupe_keys(article)
            if any(key in seen_keys for key in dedupe_keys) or len(selected) >= top_k:
                return False
            # Relevance floor for ALL sources, so irrelevant junk (memes, unrelated
            # PDFs) can't be cited or ground the forecast. Query-agnostic channels
            # (Stooq, generic RSS) clear the stricter of the two floors.
            floor = getattr(self, "_min_relevance", 0.0)
            if article.get("source_channel") in _QUERY_AGNOSTIC_CHANNELS:
                floor = max(floor, _MIN_GENERIC_RELEVANCE)
            if article.get("relevance", 1.0) < floor:
                return False
            seen_keys.update(dedupe_keys)
            selected.append(article)
            return True

        for channel in SOURCE_DIVERSITY_ORDER:
            if channel not in self._fetch_sources:
                continue
            for article in ranked:
                if article.get("source_channel") == channel and add(article):
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
        raw_lock = threading.Lock()

        def fetch_query(search_query: str, limit: int) -> None:
            articles = list(self.fetch(search_query, top_k=limit))
            for article in articles:
                article.setdefault("search_query", search_query)
            with raw_lock:
                raw.extend(articles)

        if search_queries:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_queries), 4)) as executor:
                futures = [executor.submit(fetch_query, q, max(top_k, 5)) for q in search_queries]
                concurrent.futures.wait(futures, timeout=8.0)

        for article in raw:
            if self._summarize_articles:
                article["summary"] = self.summarize(article)
            article.setdefault("search_query", search_queries[0] if search_queries else question)
        ranked = self.rank(question, raw)
        selected = self.select_diverse_sources(ranked, top_k)
        if selected:
            return selected

        # Broad searches can return candidates that are all filtered as irrelevant
        # (non-empty `ranked`, empty `selected`). Retry with forecasting-specific
        # keywords before falling back to the low-relevance ranked list --
        # returning ranked[:top_k] here would skip the retry entirely and hand
        # back whatever irrelevant junk the broad search found.
        fallback_query = _keyword_search_query(question)
        if fallback_query not in search_queries:
            previous_count = len(raw)
            fetch_query(fallback_query, max(top_k * 2, 10))
            for article in raw[previous_count:]:
                if self._summarize_articles:
                    article["summary"] = self.summarize(article)
            ranked = self.rank(question, raw)
            selected = self.select_diverse_sources(ranked, top_k)
            if selected:
                return selected
        return selected or ranked[:top_k]
