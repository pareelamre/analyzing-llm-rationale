from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# Default empirical Brier scores when live track record is unavailable or for cold models
DEFAULT_MODEL_BRIER_SCORES: Dict[str, float] = {
    "gemma-4-26b-a4b-it": 0.088,
    "council": 0.092,
    "gpt-oss-120b": 0.096,
    "qwen3-8-27b": 0.104,
    "glm-5-3": 0.108,
    "glm-5-3-flash": 0.112,
    "llama-3.3-70b-instruct": 0.115,
    "deepseek-v4-flash": 0.118,
    "minimax-m3": 0.122,
    "crowd-follow": 0.140,
}


def extract_model_briers_from_track_record(
    track_record_data: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
) -> Dict[str, float]:
    """Extract empirical model Brier scores from track_record_live data if available,
    falling back to DEFAULT_MODEL_BRIER_SCORES."""
    brier_map = dict(DEFAULT_MODEL_BRIER_SCORES)
    if not track_record_data:
        return brier_map

    # Try models_comparison table
    models_comparison = track_record_data.get("models_comparison") or []
    for row in models_comparison:
        if isinstance(row, dict):
            model = row.get("model") or row.get("model_name")
            brier = row.get("brier_score") or row.get("brier")
            if model and isinstance(brier, (int, float)) and brier > 0:
                brier_map[model] = float(brier)

    # Category specific overrides if present
    if category:
        category_briers = track_record_data.get("brier_by_category", {}).get(category, {})
        for model, brier in category_briers.items():
            if isinstance(brier, (int, float)) and brier > 0:
                brier_map[model] = float(brier)

    return brier_map


def compute_brier_weights(
    models: List[str],
    category: Optional[str] = None,
    track_record_data: Optional[Dict[str, Any]] = None,
    epsilon: float = 0.01,
) -> Dict[str, float]:
    """Compute normalized inverse-Brier weights for a given list of models.
    Higher weight is assigned to models with lower historical Brier scores."""
    brier_map = extract_model_briers_from_track_record(track_record_data, category=category)
    raw_weights: Dict[str, float] = {}

    for model in models:
        brier = brier_map.get(model, 0.12)
        # Inverse Brier score weighting
        raw_weights[model] = 1.0 / (brier + epsilon)

    total_weight = sum(raw_weights.values()) or 1.0
    return {m: round(w / total_weight, 4) for m, w in raw_weights.items()}


def aggregate_ensemble_predictions(
    member_forecasts: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    market_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate individual model probability forecasts into a calibrated ensemble.

    Parameters
    ----------
    member_forecasts: List of dicts, each with keys 'model' (str) and 'probability' (float in [0, 1]).
                      Optional keys: 'rationale', 'confidence'.
    weights: Optional mapping of model -> normalized weight.
    market_price: Optional reference market probability [0, 1].

    Returns
    -------
    Dict containing:
      - ensemble_probability: float
      - confidence_interval: [low, high]
      - dispersion_std: float
      - consensus_level: str ('high', 'moderate', 'divergent')
      - edge: Optional[float]
      - member_contributions: List[dict]
      - rationale_summary: str
    """
    if not member_forecasts:
        return {
            "ensemble_probability": 0.5,
            "confidence_interval": [0.35, 0.65],
            "dispersion_std": 0.0,
            "consensus_level": "none",
            "edge": None,
            "member_contributions": [],
            "rationale_summary": "No member forecasts provided.",
        }

    valid_members = []
    models = []
    for item in member_forecasts:
        m = item.get("model", "unknown")
        p = item.get("probability")
        if p is not None and isinstance(p, (int, float)):
            prob = max(0.001, min(0.999, float(p)))
            valid_members.append({
                "model": m,
                "probability": prob,
                "rationale": item.get("rationale", ""),
                "confidence": item.get("confidence", 0.7),
            })
            models.append(m)

    if not valid_members:
        return {
            "ensemble_probability": 0.5,
            "confidence_interval": [0.35, 0.65],
            "dispersion_std": 0.0,
            "consensus_level": "none",
            "edge": None,
            "member_contributions": [],
            "rationale_summary": "No valid model probabilities found.",
        }

    if not weights:
        weights = compute_brier_weights(models)

    # Normalize weights for the available members
    total_w = sum(weights.get(m["model"], 1.0 / len(valid_members)) for m in valid_members)
    normalized_weights = {
        m["model"]: (weights.get(m["model"], 1.0 / len(valid_members)) / (total_w or 1.0))
        for m in valid_members
    }

    # Weighted mean probability
    weighted_p = sum(m["probability"] * normalized_weights[m["model"]] for m in valid_members)
    weighted_p = round(max(0.01, min(0.99, weighted_p)), 4)

    # Weighted variance & standard deviation
    weighted_var = sum(
        normalized_weights[m["model"]] * ((m["probability"] - weighted_p) ** 2)
        for m in valid_members
    )
    std_dev = round(math.sqrt(max(0.0, weighted_var)), 4)

    # 90% confidence interval (z ~ 1.645)
    ci_low = round(max(0.01, weighted_p - 1.645 * max(std_dev, 0.03)), 4)
    ci_high = round(min(0.99, weighted_p + 1.645 * max(std_dev, 0.03)), 4)

    # Consensus level classification
    if std_dev < 0.05:
        consensus_level = "high"
    elif std_dev < 0.12:
        consensus_level = "moderate"
    else:
        consensus_level = "divergent"

    edge = None
    if market_price is not None and isinstance(market_price, (int, float)):
        edge = round(weighted_p - float(market_price), 4)

    contributions = []
    for m in valid_members:
        contributions.append({
            "model": m["model"],
            "probability": m["probability"],
            "weight": round(normalized_weights[m["model"]], 4),
            "effective_contribution": round(m["probability"] * normalized_weights[m["model"]], 4),
            "rationale": m["rationale"],
        })

    # Generate synthesis summary
    model_names = ", ".join(m["model"] for m in valid_members)
    edge_text = f" with edge of {edge:+.1%}" if edge is not None else ""
    summary = (
        f"Calibrated ensemble across {len(valid_members)} models ({model_names}) "
        f"yields a weighted forecast of {weighted_p:.1%} (90% CI: [{ci_low:.1%}, {ci_high:.1%}], "
        f"dispersion \u03c3={std_dev:.3f}, consensus: {consensus_level}){edge_text}."
    )

    return {
        "ensemble_probability": weighted_p,
        "confidence_interval": [ci_low, ci_high],
        "dispersion_std": std_dev,
        "consensus_level": consensus_level,
        "edge": edge,
        "member_contributions": contributions,
        "rationale_summary": summary,
    }
