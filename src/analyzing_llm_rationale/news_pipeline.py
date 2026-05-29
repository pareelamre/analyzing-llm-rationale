from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class NewsPipeline:
    """Fetch, summarize, and rank news articles for a forecasting question."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://llm.scads.ai/v1",
        model: str = "openai/gpt-oss-120b",
        embedding_model: str = "all-MiniLM-L6-v2",
        newsapi_key: Optional[str] = None,
    ):
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key or os.environ["SCADS_AI_API_KEY"],
            base_url=base_url,
            temperature=0.0,
            max_tokens=512,
        )
        self._embedding_model_name = embedding_model
        self._embeddings = None
        self._newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def fetch(self, query: str, top_k: int = 10) -> List[dict]:
        """Return up to top_k raw article dicts from RSS + optional NewsAPI."""
        articles: List[dict] = []

        if self._newsapi_key:
            articles.extend(self._fetch_newsapi(query, page_size=top_k))

        if len(articles) < top_k:
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

    def summarize(self, article: dict) -> str:
        """Summarize a single article using LangChain + SCADS AI LLM."""
        from langchain.schema import HumanMessage

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
            (a.get("summary") or a.get("title") or a.get("text") or "")[:512]
            for a in articles
        ]
        texts = [t if t.strip() else " " for t in texts]

        try:
            q_vec = np.array(embeddings.embed_query(question))
            doc_vecs = np.array(embeddings.embed_documents(texts))
            scores = [_cosine_similarity(q_vec, d) for d in doc_vecs]
        except Exception:
            scores = [0.0] * len(articles)

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
        raw = self.fetch(question, top_k=top_k * 2)
        for article in raw:
            article["summary"] = self.summarize(article)
        ranked = self.rank(question, raw)
        return ranked[:top_k]
