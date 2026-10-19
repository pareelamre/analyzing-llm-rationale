"""Dated Foresea track-record signal for allocating shadow research capacity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

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
    an unadjusted per-domain interval, so even positive rows are hypotheses
    for research allocation, not transferable BYOM performance evidence.
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
            if not isinstance(row, dict) or row.get("skill_significant") is not True:
                continue
            domain = row.get("domain")
            count = row.get("n")
            if not isinstance(domain, str) or not domain.strip() or len(domain) > 64:
                continue
            if isinstance(count, bool) or not isinstance(count, int) or count < 30:
                continue
            try:
                low = Decimal(str(row.get("skill_ci_low")))
                skill = Decimal(str(row.get("skill_vs_market")))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if low.is_finite() and skill.is_finite() and low > 0 and skill > 0:
                domains.add(domain.strip().lower())
        prior = ForecastResearchPrior(generated_at, sha256(raw).hexdigest(), frozenset(domains))
        return prior if domains and prior.valid_at(as_of) else None
    except (OSError, ValueError, TypeError, KeyError, InvalidOperation, RecursionError):
        return None
