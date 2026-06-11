#!/usr/bin/env python3
"""Build the public Foresea Radar artifact and commit it from GitHub Actions.

Cloud Run should only serve Radar; it should not scrape venues, scrape Reddit, or
run batch forecasts on visitor requests. This script does that heavier work on a
GitHub-hosted runner, writes ``static/radar.json``, and the workflow commits it.

Env:
  FORESEA_BASE_URL   forecast endpoint base (default https://foresea.ink)
  RADAR_LIMIT        number of public items to keep (default 10, clamped 1..25)
  RADAR_FORECASTS    fresh /predict calls for candidates not already tracked
                     (default 10, clamped 0..25)
  RADAR_OUTPUT       output artifact path (default static/radar.json)
  PREDICT_API_KEY    sent as X-API-Key if /predict is protected (optional)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import radar  # noqa: E402

BASE_URL = os.environ.get("FORESEA_BASE_URL", "https://foresea.ink").rstrip("/")
OUTPUT = Path(os.environ.get("RADAR_OUTPUT") or ROOT / "static" / "radar.json")
LIMIT = max(1, min(int(os.environ.get("RADAR_LIMIT", "10") or 10), 25))
FORECASTS = max(0, min(int(os.environ.get("RADAR_FORECASTS", "10") or 10), 25))
PREDICT_API_KEY = os.environ.get("PREDICT_API_KEY") or None

_PREDICT_TIMEOUT_S = int(os.environ.get("RADAR_PREDICT_TIMEOUT_S", "120") or 120)
_PREDICT_RETRIES = max(1, int(os.environ.get("RADAR_PREDICT_RETRIES", "3") or 3))


def _post_predict(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if PREDICT_API_KEY:
        headers["X-API-Key"] = PREDICT_API_KEY
    last_err = None
    for attempt in range(_PREDICT_RETRIES):
        req = urllib.request.Request(f"{BASE_URL}/predict", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=_PREDICT_TIMEOUT_S) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = str(exc)
        if attempt < _PREDICT_RETRIES - 1:
            time.sleep(2 ** attempt)
    print(f"  radar predict failed: {last_err}", file=sys.stderr)
    return None


def _forecast_item(item: Dict[str, Any]) -> bool:
    if not item.get("question") or not item.get("market_url") or item.get("foresea_probability") is not None:
        return False
    payload = {
        "question": item["question"],
        "attach_evidence": True,
        "evidence_top_k": 3,
        "market_platform": item.get("platform"),
        "market_url": item.get("market_url"),
        "market_ident": item.get("ident"),
        "market_outcome": item.get("outcome") or "Yes",
        # Live odds in context: pass the quote; /predict also auto-fetches fresh
        # odds from market_platform+ident when this is missing.
        "market_probability": item.get("market_probability"),
    }
    res = _post_predict(payload)
    if not res:
        return False
    analysis = res.get("market_analysis") or {}
    model_probability = analysis.get("model_probability")
    if model_probability is None:
        return False
    item["foresea_probability"] = float(model_probability)
    if analysis.get("market_probability") is not None:
        item["market_probability"] = analysis["market_probability"]
        item["probability"] = analysis["market_probability"]
    if item.get("market_probability") is not None:
        item["edge"] = item["foresea_probability"] - item["market_probability"]
    item["fresh_forecast"] = True
    item["forecast_evidence_count"] = len(res.get("evidence_sources") or [])
    return True


def main() -> int:
    raw_items, reddit = radar.collect_candidates(include_reddit=True)
    items = radar.enrich_items(raw_items, ROOT)

    # Forecast the highest-signal untracked markets first. Tracked markets already
    # have point-in-time Foresea probabilities from the committed track record.
    candidates = sorted(
        [i for i in items if i.get("market_url") and i.get("foresea_probability") is None],
        key=lambda i: (i.get("unusual_score") or 0, i.get("credibility_score") or 0),
        reverse=True,
    )
    forecasted = 0
    for item in candidates[:FORECASTS]:
        if _forecast_item(item):
            forecasted += 1

    # Recompute tags/scores after fresh forecasts add edge/fresh_forecast fields.
    items = radar.enrich_items(items, ROOT)
    payload = radar.render_radar_payload(
        items,
        reddit,
        limit=LIMIT,
        generated_by="github_action",
    )
    payload["fresh_forecasts"] = forecasted
    payload["forecast_budget"] = FORECASTS

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "items": len(payload.get("items") or []),
        "reddit_discussions": len(payload.get("reddit_discussions") or []),
        "fresh_forecasts": forecasted,
        "output": str(OUTPUT),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
