"""Dated Foresea track-record signal for allocating shadow research capacity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from ..entity_tagger import DOMAINS
from ..forecast_evaluation import has_qualified_market_skill

_MAX_AGE = timedelta(days=7)
_MAX_BYTES = 16_000_000


@dataclass(frozen=True)
class ForecastResearchPrior:
    generated_at: datetime
    artifact_hash: str
    validated_domains: frozenset[str]

    def valid_at(self, as_of: datetime) -> bool:
        return (
            as_of.tzinfo is not None
            and as_of.utcoffset() is not None
            and self.generated_at < as_of
            and as_of - self.generated_at <= _MAX_AGE
        )


def load_forecast_research_prior(
    path: Path, *, as_of: datetime,
) -> ForecastResearchPrior | None:
    """Read a research hint, never a sizing calibration.

    Foresea reports skill as market Brier minus model Brier. Its domain CI is
    a corrected interval over market means. Legacy per-snapshot confidence
    cannot prioritize research. Positive rows remain research hypotheses,
    not transferable BYOM performance or executable-return evidence.
    """
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("source") != "live":
            return None
        generated_at = datetime.fromisoformat(str(payload["generated_at"]))
        if generated_at.tzinfo is None:
            return None
        generated_at = generated_at.astimezone(timezone.utc)
        rows = payload.get("by_domain")
        if not isinstance(rows, list):
            return None
        domains: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not has_qualified_market_skill(
                row, minimum_comparisons=max(len(DOMAINS), len(rows)),
            ):
                continue
            domain = row.get("domain")
            if not isinstance(domain, str) or not domain.strip() or len(domain) > 64:
                continue
            domains.add(domain.strip().lower())
        prior = ForecastResearchPrior(generated_at, sha256(raw).hexdigest(), frozenset(domains))
        return prior if domains and prior.valid_at(as_of) else None
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return None
