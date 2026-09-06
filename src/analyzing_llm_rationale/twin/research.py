"""Bounded, evidence-only research loop that cannot produce an execution command."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence

from .budget import call_with_budget
from .models import (
    Forecast,
    PassDecision,
    Proposal,
    ProposalAction,
    RejectionReason,
)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def research_forecast(
    provider: Callable[[str], Mapping[str, Any]], *, budget: Any, budget_key: str, budget_policy: Any,
    reservation_id: str, instrument_id: str, snapshot_id: str, as_of: datetime, evidence: Sequence[Mapping[str, Any]],
    model_id: str, prompt: str,
) -> tuple[Forecast | None, Proposal]:
    """Return a validated forecast or a structured PASS; never returns a trade intent."""
    valid = [item for item in evidence if isinstance(item, Mapping) and item.get("id") and item.get("observed_at") and str(item["observed_at"]) <= as_of.isoformat()]
    if not valid:
        return None, _pass(snapshot_id, "missing-evidence", RejectionReason.PASS_INCOMPLETE_DATA, as_of)
    bounded = [{"id": str(item["id"]), "text": str(item.get("text") or "")[:1200]} for item in valid[:8]]
    payload = json.dumps({"instruction": "Return JSON only: p_yes, uncertainty_low, uncertainty_high, evidence_ids.", "market": instrument_id, "as_of": as_of.isoformat(), "evidence": bounded}, sort_keys=True)
    try:
        response = call_with_budget(budget, reservation_id, key=budget_key, estimated_usd=Decimal("0"), estimated_tokens=2000, policy=budget_policy, operation=lambda: provider(payload))
        raw = response.get("content") if isinstance(response, Mapping) else None
        parsed = json.loads(raw) if isinstance(raw, str) else response
        if not isinstance(parsed, Mapping):
            raise ValueError("response is not an object")
        evidence_ids = tuple(str(value) for value in parsed.get("evidence_ids") or ())
        if not evidence_ids or not set(evidence_ids).issubset({item["id"] for item in bounded}):
            raise ValueError("response cited unknown evidence")
        forecast_id = f"forecast-{_hash(instrument_id + snapshot_id + as_of.isoformat())[:24]}"
        forecast = Forecast(forecast_id, instrument_id, Decimal(str(parsed["p_yes"])), None, "insufficient", Decimal(str(parsed["uncertainty_low"])), Decimal(str(parsed["uncertainty_high"])), evidence_ids, as_of, as_of + timedelta(hours=6), _hash(model_id), _hash(prompt), "foresea-edge-v1", "calibration-v1", "prospective-v1", as_of)
        proposal = Proposal(f"proposal-{_hash(forecast_id)[:24]}", forecast.id, snapshot_id, ProposalAction.HOLD, (), evidence_ids, None, None, as_of)
        return forecast, proposal
    except Exception:
        return None, _pass(snapshot_id, "invalid-research", RejectionReason.PASS_INVALID_PROPOSAL, as_of)


def _pass(snapshot_id: str, detail: str, reason: RejectionReason, now: datetime) -> Proposal:
    forecast_id = f"pass-forecast-{_hash(snapshot_id + detail)[:24]}"
    return Proposal(f"proposal-{_hash(snapshot_id + detail)[:24]}", forecast_id, snapshot_id, ProposalAction.PASS, (reason,), (), None, PassDecision(reason, detail), now)
