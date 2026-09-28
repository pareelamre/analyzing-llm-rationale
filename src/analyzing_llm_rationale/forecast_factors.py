from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def extract_factors_and_catalysts(
    rationale: str,
    probability: Optional[float] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """Decompose a written forecast rationale into structured bull factors,
    bear factors, and key catalysts.

    Parameters
    ----------
    rationale: Model written explanation
    probability: Forecast probability (0.0 to 1.0)
    question: The original market question

    Returns
    -------
    Dict with:
      - bull_factors: List of dicts (factor, impact, weight)
      - bear_factors: List of dicts (factor, impact, weight)
      - key_catalysts: List of dicts (event, trigger_type)
    """
    if not rationale:
        return {
            "bull_factors": [],
            "bear_factors": [],
            "key_catalysts": [],
        }

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rationale) if s.strip()]
    bull_factors: List[Dict[str, Any]] = []
    bear_factors: List[Dict[str, Any]] = []
    key_catalysts: List[Dict[str, Any]] = []

    # Common sentiment cues
    bull_cues = [
        "increase", "higher", "growth", "accelerat", "positive", "strong",
        "likely", "momentum", "bull", "advanc", "gain", "expand", "boost",
        "favor", "pass", "approv", "adopt", "surge", "outperform",
    ]
    bear_cues = [
        "delay", "risk", "decline", "slowdown", "negative", "weak",
        "unlikely", "bear", "drop", "fell", "barrier", "oppos", "hinder",
        "reject", "fail", "headwind", "uncertain", "obstacle", "stagnat",
    ]
    catalyst_cues = [
        "deadline", "scheduled", "meeting", "vote", "announcement", "decision",
        "election", "report", "earnings", "release", "summit", "conference",
        "by date", "before", "in Q", "in 202", "launch",
    ]

    prob_val = probability if probability is not None else 0.5

    for s in sentences:
        s_lower = s.lower()
        bull_score = sum(1 for cue in bull_cues if cue in s_lower)
        bear_score = sum(1 for cue in bear_cues if cue in s_lower)
        is_catalyst = any(cue in s_lower for cue in catalyst_cues)

        if is_catalyst:
            key_catalysts.append({
                "event": s[:180],
                "trigger_type": "scheduled_event" if "schedul" in s_lower or "date" in s_lower else "catalyst",
            })

        if bull_score > bear_score:
            impact = "high" if bull_score >= 2 or prob_val >= 0.70 else "moderate"
            bull_factors.append({
                "factor": s[:200],
                "impact": impact,
                "weight": round(0.5 + (0.1 * bull_score), 2),
            })
        elif bear_score > bull_score:
            impact = "high" if bear_score >= 2 or prob_val <= 0.30 else "moderate"
            bear_factors.append({
                "factor": s[:200],
                "impact": impact,
                "weight": round(0.5 + (0.1 * bear_score), 2),
            })
        else:
            # Ambiguous or balanced sentences
            if prob_val >= 0.55:
                bull_factors.append({
                    "factor": s[:200],
                    "impact": "low",
                    "weight": 0.4,
                })
            else:
                bear_factors.append({
                    "factor": s[:200],
                    "impact": "low",
                    "weight": 0.4,
                })

    return {
        "bull_factors": bull_factors[:5],
        "bear_factors": bear_factors[:5],
        "key_catalysts": key_catalysts[:4],
    }
