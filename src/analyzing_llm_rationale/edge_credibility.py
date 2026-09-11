"""Foresea Market Edge Credibility & Verification Engine.

Audits and verifies whether detected statistical market edges (model-vs-market probability gaps)
are credible, actionable, and grounded in verifiable reality, or artifacts of:
- Illiquid / spoofed orderbooks
- Ambiguous or subjective resolution criteria
- Stale or missing evidence
- Outdated / expired horizon dates
- Hallucinated extreme probability divergences

Provides structured scoring:
- credibility_score (0.0 to 1.0)
- credibility_grade ("A", "B", "C")
- credibility_flags (list of positive & cautionary signals)
- is_credible (bool)
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List


def _first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    """First key that is present and not None -- `or` would skip a real 0.

    market_volume is 0.0 on a market nobody is quoting, which is exactly the
    case the liquidity penalty exists for. An `or` chain treats that as
    absent and falls through.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _evidence_count(opp: Dict[str, Any]) -> int:
    """How many evidence items back this opportunity.

    The edge board publishes a count, not the items. Reading only `evidence`
    meant every board row looked unevidenced.
    """
    evidence = opp.get("evidence")
    if isinstance(evidence, list):
        return len(evidence)
    raw = opp.get("evidence_count")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _parse_dt(val: Any) -> datetime | None:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if not val or not isinstance(val, str):
        return None
    cleaned = val.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def audit_edge_opportunity(opp: Dict[str, Any]) -> Dict[str, Any]:
    """Audit a single market edge opportunity and return credibility metadata."""
    question = str(opp.get("question") or opp.get("title") or "").strip()
    model_p = opp.get("model_probability")
    mkt_p = opp.get("market_probability")
    resolution_criteria = str(opp.get("resolution_criteria") or opp.get("description") or "").strip()
    # market_volume / market_liquidity are what the edge board actually
    # publishes; the three names below appear on none of its rows, so the
    # liquidity assessment never ran on real data.
    volume = _first_present(
        opp, "volume", "volume_usd", "open_interest", "market_volume", "market_liquidity"
    )
    evidence_items = _evidence_count(opp)
    horizon = str(opp.get("horizon") or opp.get("lead_bucket") or "").lower()

    flags: List[str] = []
    score: float = 1.0

    # 1. Edge & Probability Sanity Check
    if model_p is None or mkt_p is None:
        return {
            "credibility_score": 0.0,
            "credibility_grade": "C",
            "credibility_flags": ["missing_probability_data"],
            "is_credible": False,
            "audit_summary": "Missing probability values for comparison.",
        }

    try:
        model_p_val = float(model_p)
        mkt_p_val = float(mkt_p)
    except (ValueError, TypeError):
        return {
            "credibility_score": 0.0,
            "credibility_grade": "C",
            "credibility_flags": ["invalid_probability_format"],
            "is_credible": False,
            "audit_summary": "Invalid probability format.",
        }

    edge = abs(model_p_val - mkt_p_val)

    # Edge sanity: Extreme edge (> 50%) without multiple sources is penalized
    if edge > 0.50:
        if evidence_items < 2:
            score -= 0.25
            flags.append("extreme_edge_sparse_evidence")
        else:
            flags.append("high_discrepancy_well_evidenced")
    elif edge >= 0.08:
        flags.append("actionable_edge_threshold_met")
    else:
        flags.append("narrow_edge")

    # 2. Evidence Grounding Audit
    if evidence_items > 0:
        score += 0.10
        flags.append(f"grounded_{evidence_items}_evidence_items")
    else:
        # Absence of explicit evidence in payload
        score -= 0.15
        flags.append("sparse_retrieved_evidence")

    # 3. Resolution Criteria Clarity Audit
    if resolution_criteria:
        crit_len = len(resolution_criteria)
        if crit_len > 40:
            flags.append("verifiable_resolution_criteria")
            score += 0.05
        else:
            flags.append("minimal_resolution_criteria")
    else:
        # Check if question itself is self-contained (e.g. "Will X reach Y by Date?")
        has_date = bool(re.search(r"\b(202[4-9]|January|February|March|April|May|June|July|August|September|October|November|December)\b", question, re.I))
        if has_date:
            flags.append("self_contained_timeline_in_title")
        else:
            score -= 0.10
            flags.append("missing_explicit_resolution_rules")

    # 4. Expiry / Horizon Audit
    has_past_due = False
    now_utc = datetime.now(timezone.utc)
    resolve_raw = _first_present(opp, "resolve_time", "close_time", "resolution_date")
    parsed_resolve = _parse_dt(resolve_raw)
    if parsed_resolve is not None and parsed_resolve <= now_utc:
        has_past_due = True
    else:
        date_matches = re.findall(r"\b(20[1-2][0-9])\b", question)
        current_year = now_utc.year
        for yr in date_matches:
            if int(yr) < current_year:
                has_past_due = True
                break

    if has_past_due:
        score -= 0.60
        flags.append("potential_past_due_market")
    elif "day" in horizon or "week" in horizon or "month" in horizon:
        flags.append(f"horizon_{horizon}")

    # 5. Liquidity, Spread & Volume Assessment
    market_bid = _first_present(opp, "market_bid", "bid")
    market_ask = _first_present(opp, "market_ask", "ask")
    if market_bid is not None and market_ask is not None:
        try:
            bid_val = float(market_bid)
            ask_val = float(market_ask)
            if ask_val >= bid_val:
                spread = ask_val - bid_val
                if spread >= 0.25:
                    score -= 0.30
                    flags.append("wide_bid_ask_spread")
                elif spread >= 0.12:
                    score -= 0.15
                    flags.append("moderate_bid_ask_spread")
                else:
                    flags.append("tight_spread")
        except (ValueError, TypeError):
            pass

    if volume is not None:
        try:
            vol_val = float(volume)
            if vol_val >= 5000:
                score += 0.10
                flags.append("healthy_market_liquidity")
            elif vol_val < 200:
                score -= 0.20
                flags.append("low_liquidity_spread_risk")
        except (ValueError, TypeError):
            pass

    # Normalize score between 0.0 and 1.0
    final_score = max(0.0, min(1.0, round(score, 2)))

    # Compute Grade
    if final_score >= 0.80:
        grade = "A"
        is_credible = True
        summary = "High credibility: Grounded evidence, clear resolution parameters, robust statistical edge."
    elif final_score >= 0.60:
        grade = "B"
        is_credible = True
        summary = "Moderate credibility: Tradable edge with standard confidence parameters."
    else:
        grade = "C"
        is_credible = False
        summary = "Caution: Higher uncertainty or limited evidence/liquidity; recommend manual verification."

    return {
        "credibility_score": final_score,
        "credibility_grade": grade,
        "credibility_flags": flags,
        "is_credible": is_credible,
        "audit_summary": summary,
    }


def audit_edge_board(opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Audit a list of edge opportunities and attach credibility scores."""
    audited = []
    for opp in opportunities:
        opp_copy = dict(opp)
        audit_res = audit_edge_opportunity(opp_copy)
        opp_copy.update(audit_res)
        audited.append(opp_copy)
    return audited
