from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence
from urllib.parse import urlencode

import numpy as np

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
]

DEFAULT_FETCH_SOURCES = ("newsapi", "gdelt", "google-news", "rss")


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

        if self._newsapi_key and "newsapi" in self._fetch_sources:
            articles.extend(self._fetch_newsapi(query, page_size=top_k))

        if "gdelt" in self._fetch_sources and len(articles) < top_k:
            articles.extend(self._fetch_gdelt(query, limit=top_k * 2))

        if "google-news" in self._fetch_sources and len(articles) < top_k:
            articles.extend(self._fetch_google_news(query, limit=top_k * 2))

        if "rss" in self._fetch_sources and len(articles) < top_k:
            articles.extend(self._fetch_rss(limit=top_k * 2))

        seen_urls: set = set()
        unique: List[dict] = []
        for a in articles:
            url = a.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(a)

        return unique[:top_k * 2]

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
                })
            return articles
        except Exception:
            return []

    def _fetch_google_news(self, query: str, limit: int = 20) -> List[dict]:
        try:
            import feedparser
        except ImportError:
            return []

        params = urlencode({
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        })
        feed_url = f"https://news.google.com/rss/search?{params}"

        try:
            feed = feedparser.parse(feed_url)
        except Exception:
            return []

        articles: List[dict] = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title") or "",
                "url": entry.get("link") or "",
                "publish_date": entry.get("published") or "",
                "text": entry.get("summary") or "",
                "summary": entry.get("summary") or "",
                "source": "Google News",
            })
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
                    })
            except Exception:
                continue
        return articles

    def plan_search_query(self, question: str) -> str:
        """Use a small LangChain planner step to turn a forecast into a news query."""
        if not self._use_query_planner or self._llm is None:
            return question

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You write concise web-news search queries for binary forecasting questions.",
                ),
                (
                    "user",
                    "Forecasting question:\n{question}\n\n"
                    "Return one search query, no quotes, no explanation. "
                    "Preserve important entities, dates, and event terms.",
                ),
            ])
            chain = prompt | self._llm | StrOutputParser()
            planned = chain.invoke({"question": question}).strip()
            return planned[:200] or question
        except Exception:
            return question

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
            scores = [_lexical_relevance(question, text) for text in texts]
        else:
            try:
                q_vec = np.array(embeddings.embed_query(question))
                doc_vecs = np.array(embeddings.embed_documents(texts))
                scores = [_cosine_similarity(q_vec, d) for d in doc_vecs]
            except Exception:
                scores = [_lexical_relevance(question, text) for text in texts]

        ranked = sorted(
            zip(scores, articles),
            key=lambda x: x[0],
            reverse=True,
        )
        for score, article in ranked:
            article["relevance_score"] = round(score, 4)
        return [a for _, a in ranked]

    def fetch_summarize_rank(
        self, question: str, top_k: int = 5
    ) -> List[dict]:
        """Full pipeline: fetch → summarize → rank → return top_k."""
        search_query = self.plan_search_query(question)
        raw = self.fetch(search_query, top_k=top_k * 2)
        for article in raw:
            if self._summarize_articles:
                article["summary"] = self.summarize(article)
            article["search_query"] = search_query
        ranked = self.rank(question, raw)
        return ranked[:top_k]
