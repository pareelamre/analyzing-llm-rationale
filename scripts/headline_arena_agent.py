"""Headline Arena Autonomous Forecasting Agent for Foresea.

Orchestrates daily macro asset and price-event forecasting on Headline Arena
using Foresea's 3-model deliberation Council (`gemma-4-26b-a4b-it`, `gpt-oss-120b`,
`qwen3-8-27b`) with automatic fallback to single-model `gemma-4-26b-a4b-it`.
"""

import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("headline_arena_agent")

BASE_URL = os.environ.get("HEADLINE_ARENA_URL", "https://headlinearena.com/api/v1")
FORESEA_API_URL = os.environ.get("FORESEA_API_URL", "https://foresea.ink")

AGENT_ID = os.environ.get("HEADLINE_ARENA_AGENT_ID", "agt_56952984bb35")
CLIENT_SECRET = os.environ.get("HEADLINE_ARENA_CLIENT_SECRET")

# If credentials file exists locally, load defaults from it
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "ha_reg_response.json")
if not CLIENT_SECRET and os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            AGENT_ID = cfg.get("agent_id", AGENT_ID)
            CLIENT_SECRET = cfg.get("client_secret", CLIENT_SECRET)
    except Exception as exc:
        logger.warning("Could not load local ha_reg_response.json: %s", exc)


def get_access_token() -> str:
    """Request a 60-minute bearer access token from Headline Arena."""
    if not CLIENT_SECRET:
        raise RuntimeError("HEADLINE_ARENA_CLIENT_SECRET is required to authenticate.")

    url = f"{BASE_URL}/agent/auth/token"
    payload = {
        "grant_type": "client_credentials",
        "agent_id": AGENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Foresea-Agent/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as res:
        data = json.loads(res.read().decode("utf-8"))
        logger.info("Successfully authenticated with Headline Arena. Token valid for %ss", data.get("expires_in"))
        return data["access_token"]


def get_active_challenges(token: str) -> List[Dict[str, Any]]:
    """Fetch active challenges eligible for predictions."""
    url = f"{BASE_URL}/eval/challenges/active"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "Foresea-Agent/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
            if isinstance(data, dict):
                return data.get("challenges") or data.get("items") or []
            if isinstance(data, list):
                return data
            return []
    except Exception as exc:
        logger.error("Failed to query active challenges: %s", exc)
        return []


def query_foresea_forecast(question: str, model: str = "council") -> Tuple[float, str]:
    """Call Foresea /predict endpoint. Returns (model_probability, rationale)."""
    url = f"{FORESEA_API_URL}/predict"
    payload = {
        "question": question,
        "model": model,
        "attach_evidence": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Foresea-HeadlineArena-Runner/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read().decode("utf-8"))
        analysis = data.get("market_analysis") or {}
        prob = analysis.get("model_probability")
        if prob is None:
            prob = data.get("confidence", 0.5)
        rationale = data.get("rationale") or analysis.get("rationale") or "Deliberated rationale provided by Foresea."
        return float(prob), str(rationale)


def generate_forecast_with_fallback(question: str) -> Tuple[float, str, str]:
    """Try 3-model Council first; fall back to Gemma-4-26B if Council fails or times out."""
    try:
        logger.info("Forecasting with 3-model Council deliberation: '%s'", question)
        prob, rationale = query_foresea_forecast(question, model="council")
        return prob, rationale, "council"
    except Exception as exc:
        logger.warning("Council deliberation failed or timed out (%s). Falling back to gemma-4-26b-a4b-it...", exc)
        try:
            prob, rationale = query_foresea_forecast(question, model="gemma-4-26b-a4b-it")
            return prob, rationale, "gemma-4-26b-a4b-it"
        except Exception as gemma_exc:
            logger.error("Fallback to Gemma also failed: %s", gemma_exc)
            raise gemma_exc


def map_probability_to_prediction(prob: float, rationale: str) -> Dict[str, Any]:
    """Map binary probability (0.0 to 1.0) to ternary (bullish/bearish/neutral) with confidence."""
    if prob > 0.55:
        direction = "bullish"
        conf = min(0.95, prob)
    elif prob < 0.45:
        direction = "bearish"
        conf = min(0.95, 1.0 - prob)
    else:
        direction = "neutral"
        conf = max(0.40, 1.0 - abs(prob - 0.5) * 2)

    summary_first_sentence = rationale.split("\n")[0][:300]
    if len(summary_first_sentence) < 30:
        summary = f"Foresea multi-agent deliberation indicates a {direction} lean with {round(conf*100)}% model confidence."
    else:
        summary = f"{summary_first_sentence.strip()} Foresea assigns {round(conf*100)}% confidence to {direction} outcome."

    return {
        "direction": direction,
        "confidence": round(conf, 3),
        "reasoning": rationale[:7900],
        "summary": summary[:490],
    }


def submit_prediction(token: str, challenge_id: str, payload: Dict[str, Any]) -> bool:
    """Submit prediction to Headline Arena."""
    url = f"{BASE_URL}/eval/challenges/{challenge_id}/predict"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Foresea-Agent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            logger.info("Challenge %s prediction submitted successfully (Status %s)", challenge_id, res.status)
            return True
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="ignore")
        logger.error("Failed to submit prediction for %s: %s %s - %s", challenge_id, err.code, err.reason, body)
        return False
    except Exception as exc:
        logger.error("Unexpected error submitting prediction: %s", exc)
        return False


def run_cycle():
    """Run one discovery and prediction cycle across open challenges."""
    logger.info("Starting Headline Arena forecast cycle for agent %s...", AGENT_ID)
    try:
        token = get_access_token()
    except Exception as exc:
        logger.error("Cannot authenticate: %s", exc)
        return

    challenges = get_active_challenges(token)
    logger.info("Found %d active challenges.", len(challenges))

    for ch in challenges:
        cid = ch.get("id") or ch.get("challenge_id")
        q = ch.get("question") or ch.get("title")
        asset = ch.get("asset", "")
        if not cid or not q:
            continue

        deadline = ch.get("deadline")
        logger.info("Evaluating challenge %s [%s]: '%s' (deadline: %s)", cid, asset, q, deadline)

        try:
            prob, rationale, engine_used = generate_forecast_with_fallback(q)
            pred_payload = map_probability_to_prediction(prob, rationale)
            logger.info(
                "Prediction generated via %s: direction=%s, confidence=%.3f",
                engine_used,
                pred_payload["direction"],
                pred_payload["confidence"],
            )
            submit_prediction(token, cid, pred_payload)
        except Exception as exc:
            logger.error("Error processing challenge %s: %s", cid, exc)

    logger.info("Forecast cycle completed.")


if __name__ == "__main__":
    run_cycle()
