"""Bounded acquisition of public evidence for one frozen market capture."""
from __future__ import annotations

import html
import ipaddress
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from .models import Instrument
from .research_gateway import PublicEvidence

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
evidence_acquisitions = metrics.get_meter(__name__).create_counter(
    "twin.evidence.acquisitions", unit="1",
)


class PublicEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicEvidencePolicy:
    max_evidence: int = 5
    max_age_days: int = 30
    min_relevance: float = 0.25

    def __post_init__(self) -> None:
        if not 1 <= self.max_evidence <= 8:
            raise PublicEvidenceError("public evidence limit is invalid")
        if not 1 <= self.max_age_days <= 365:
            raise PublicEvidenceError("public evidence age is invalid")
        if not 0 <= self.min_relevance <= 1:
            raise PublicEvidenceError("public evidence relevance is invalid")


class PublicArticleGateway(Protocol):
    def search(self, query: str, *, limit: int) -> Sequence[Mapping[str, Any]]: ...


class NewsPipelinePublicArticleGateway:
    """Reuse Foresea's news fetchers without invoking a query or summary LLM."""

    def __init__(self, *, min_relevance: float = 0.25) -> None:
        from ..news_pipeline import NewsPipeline

        self._pipeline = NewsPipeline(
            use_query_planner=False,
            summarize_articles=False,
            use_embeddings=False,
            min_relevance=min_relevance,
            fetch_sources=("web", "gdelt", "google-news", "bing-news"),
        )

    def search(self, query: str, *, limit: int) -> Sequence[Mapping[str, Any]]:
        return self._pipeline.fetch_summarize_rank(query, top_k=limit)


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PublicEvidenceError("public evidence clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _public_url(raw: Any) -> str | None:
    value = str(raw or "").strip()
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return None
        if host == "localhost" or host.endswith((".localhost", ".local")):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if not address.is_global:
                return None
        return value[:256]
    except ValueError:
        return None


def _published_at(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _plain_text(article: Mapping[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    body = str(article.get("summary") or article.get("text") or "").strip()
    combined = title if not body else f"{title}. {body}" if title else body
    combined = html.unescape(re.sub(r"<[^>]{0,500}>", " ", combined))
    return re.sub(r"\s+", " ", combined).strip()[:1200]


@tracer.start_as_current_span("twin.evidence.acquire")
def acquire_public_evidence(
    gateway: PublicArticleGateway, *, instrument: Instrument, now: datetime,
    policy: PublicEvidencePolicy | None = None,
) -> tuple[PublicEvidence, ...]:
    """Acquire one immutable, recent, public evidence set or fail closed."""
    span = trace.get_current_span()
    policy = policy or PublicEvidencePolicy()
    retrieved_at = _utc(now)
    query = str(instrument.display_title or "").strip()
    span.set_attributes({
        "market.venue": instrument.venue,
        "evidence.limit": policy.max_evidence,
    })
    if not query or len(query) > 500:
        evidence_acquisitions.add(1, {"outcome": "rejected", "venue": instrument.venue})
        raise PublicEvidenceError("market title is unavailable for public research")
    try:
        rows = gateway.search(query, limit=policy.max_evidence)
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR))
        evidence_acquisitions.add(1, {"outcome": "failure", "venue": instrument.venue})
        logger.warning("Public evidence acquisition failed", extra={"venue": instrument.venue})
        raise PublicEvidenceError("public evidence provider is unavailable") from exc
    evidence: list[PublicEvidence] = []
    seen: set[str] = set()
    oldest = retrieved_at - timedelta(days=policy.max_age_days)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            relevance = float(row.get("relevance"))
        except (TypeError, ValueError):
            continue
        if not 0 <= relevance <= 1 or relevance < policy.min_relevance:
            continue
        source_id = _public_url(row.get("url"))
        text = _plain_text(row)
        published = _published_at(row.get("publish_date") or row.get("published_at"))
        if not source_id or not text or published is None or not oldest <= published <= retrieved_at:
            continue
        identity = "evidence-" + sha256(
            f"{source_id}\n{published.isoformat()}\n{text}".encode("utf-8")
        ).hexdigest()[:24]
        if identity in seen:
            continue
        seen.add(identity)
        evidence.append(PublicEvidence(
            identity, source_id, text, published, retrieved_at,
        ))
        if len(evidence) >= policy.max_evidence:
            break
    span.set_attribute("evidence.count", len(evidence))
    if not evidence:
        evidence_acquisitions.add(1, {"outcome": "empty", "venue": instrument.venue})
        raise PublicEvidenceError("no recent public evidence passed validation")
    evidence_acquisitions.add(1, {"outcome": "success", "venue": instrument.venue})
    return tuple(evidence)
