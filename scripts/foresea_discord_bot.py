"""Foresea Discord Signal & Forecasting Bot.

A modular Discord client and webhook broadcaster that brings Foresea's
calibrated AI forecasting and prediction-market edge feeds to Discord servers.

Supports:
- Webhook Broadcasts: Post rich Discord Embeds to announcement channels on schedule.
- Command Processing: Process slash commands or message commands (!forecast, !edge, !track).

Usage:
    # Post an Edge Board alert to a Discord Webhook URL:
    python scripts/foresea_discord_bot.py --webhook-url "https://discord.com/api/webhooks/..." --post-edge

    # Run as interactive bot:
    export DISCORD_BOT_TOKEN="your_bot_token"
    python scripts/foresea_discord_bot.py
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("foresea-discord-bot")

DEFAULT_FORESEA_URL = "https://foresea.ink"


class ForeseaDiscordClient:
    """Client for generating Discord Embeds from Foresea data and posting them."""

    def __init__(
        self,
        foresea_base_url: str = DEFAULT_FORESEA_URL,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = foresea_base_url.rstrip("/")
        self.session = session or requests.Session()

    def build_forecast_embed(self, question: str) -> Dict[str, Any]:
        """Query Foresea POST /predict and format a rich Discord Embed."""
        url = f"{self.base_url}/predict"
        resp = self.session.post(url, json={"question": question}, timeout=60)
        if resp.status_code != 200:
            return {
                "title": "❌ Forecast Error",
                "description": f"Foresea API returned status HTTP {resp.status_code}.",
                "color": 0xFF0000,
            }
        data = resp.json()
        prob = data.get("predicted_probability")
        ans = data.get("predicted_answer", "Unknown")
        rationale = data.get("rationale", "").strip()
        evidence = data.get("evidence", [])

        prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
        color = 0x00FF88 if ans.upper() == "YES" else 0xFF4444

        embed: Dict[str, Any] = {
            "title": "🔮 Foresea Calibrated Forecast",
            "description": f"**Question:** {question}\n\n**Predicted Probability:** `{prob_str}` ({ans.upper()})",
            "color": color,
            "url": f"{self.base_url}",
            "fields": [],
            "footer": {"text": "Foresea Forecasting Engine • foresea.ink"},
        }

        if rationale:
            embed["fields"].append({
                "name": "📝 Rationale",
                "value": rationale[:500] + ("..." if len(rationale) > 500 else ""),
                "inline": False,
            })

        if evidence and isinstance(evidence, list):
            ev_lines = []
            for ev in evidence[:3]:
                title = ev.get("title") or ev.get("snippet", "")
                if title:
                    ev_lines.append(f"• {title[:80]}")
            if ev_lines:
                embed["fields"].append({
                    "name": "📰 Key Evidence",
                    "value": "\n".join(ev_lines),
                    "inline": False,
                })

        return embed

    def build_edge_board_embed(self, limit: int = 5) -> Dict[str, Any]:
        """Query Foresea GET /edge-board and format top mispriced opportunities."""
        url = f"{self.base_url}/edge-board"
        resp = self.session.get(url, params={"limit": limit}, timeout=15)
        if resp.status_code != 200:
            return {
                "title": "❌ Edge Board Error",
                "description": f"Foresea API returned status HTTP {resp.status_code}.",
                "color": 0xFF0000,
            }
        data = resp.json()
        if isinstance(data, dict):
            opps = data.get("opportunities") or data.get("edge_board") or []
        elif isinstance(data, list):
            opps = data
        else:
            opps = []

        embed: Dict[str, Any] = {
            "title": "⚡ Foresea Edge Board — Top Opportunities",
            "description": "Prediction market opportunities ranked by model-vs-market edge:",
            "color": 0x00AAFF,
            "url": f"{self.base_url}/#radar",
            "fields": [],
            "footer": {"text": "Foresea Radar Desk • foresea.ink"},
        }

        for idx, opp in enumerate(opps[:limit], 1):
            title = opp.get("question") or opp.get("title") or "Market"
            platform = opp.get("platform", "Venue")
            mkt_p = opp.get("market_probability")
            model_p = opp.get("model_probability")
            edge = opp.get("edge") or (abs(model_p - mkt_p) if model_p is not None and mkt_p is not None else None)
            rec = opp.get("recommendation") or ("BUY YES" if (model_p or 0) > (mkt_p or 0) else "BUY NO")

            mkt_str = f"{mkt_p*100:.0f}%" if mkt_p is not None else "?"
            model_str = f"{model_p*100:.0f}%" if model_p is not None else "?"
            edge_str = f"{edge*100:+.1f}%" if edge is not None else ""

            embed["fields"].append({
                "name": f"{idx}. [{platform}] {title[:60]}",
                "value": f"**Rec:** `{rec}` | **Market:** `{mkt_str}` → **Foresea:** `{model_str}` (**{edge_str} edge**)",
                "inline": False,
            })

        return embed

    def build_track_record_embed(self) -> Dict[str, Any]:
        """Query Foresea GET /track-record and format calibration status."""
        url = f"{self.base_url}/track-record"
        resp = self.session.get(url, timeout=15)
        if resp.status_code != 200:
            return {
                "title": "❌ Track Record Error",
                "description": f"Foresea API returned status HTTP {resp.status_code}.",
                "color": 0xFF0000,
            }
        data = resp.json()
        n_resolved = data.get("n_snapshots_resolved", 0)
        overall = data.get("overall") or {}
        brier = overall.get("mean_brier_score")
        acc = overall.get("accuracy")
        skill = overall.get("skill_vs_market")

        brier_str = f"{brier:.4f}" if brier is not None else "N/A"
        acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
        skill_str = f"{skill:+.4f}" if skill is not None else "N/A"

        return {
            "title": "📈 Foresea Live Track Record",
            "description": "Verified model performance and calibration metrics:",
            "color": 0x9B59B6,
            "url": f"{self.base_url}/#track-record",
            "fields": [
                {"name": "Resolved Forecasts", "value": f"`{n_resolved}`", "inline": True},
                {"name": "Mean Brier Score", "value": f"`{brier_str}`", "inline": True},
                {"name": "Directional Accuracy", "value": f"`{acc_str}`", "inline": True},
                {"name": "Skill vs Market", "value": f"`{skill_str}`", "inline": True},
            ],
            "footer": {"text": "Transparent Forecaster Calibration • foresea.ink"},
        }

    def post_to_webhook(self, webhook_url: str, embed: Dict[str, Any], content: str = "") -> bool:
        """Post a formatted embed to a Discord Webhook URL."""
        payload: Dict[str, Any] = {"embeds": [embed]}
        if content:
            payload["content"] = content
        try:
            resp = self.session.post(webhook_url, json=payload, timeout=10)
            return resp.status_code in (200, 204)
        except Exception as exc:
            logger.error("Failed to post to Discord webhook: %s", exc)
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresea Discord Bot & Webhook Poster")
    parser.add_argument("--webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL", ""), help="Discord Webhook URL")
    parser.add_argument("--base-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL), help="Foresea API URL")
    parser.add_argument("--post-edge", action="store_true", help="Post current top Edge Board opportunities to webhook")
    parser.add_argument("--post-track", action="store_true", help="Post current track record to webhook")
    parser.add_argument("--forecast", type=str, default="", help="Question to forecast and post to webhook")
    args = parser.parse_args()

    client = ForeseaDiscordClient(foresea_base_url=args.base_url)

    if args.webhook_url:
        if args.post_edge:
            embed = client.build_edge_board_embed()
            ok = client.post_to_webhook(args.webhook_url, embed)
            logger.info("Posted Edge Board to Discord Webhook: %s", "Success" if ok else "Failed")
        elif args.post_track:
            embed = client.build_track_record_embed()
            ok = client.post_to_webhook(args.webhook_url, embed)
            logger.info("Posted Track Record to Discord Webhook: %s", "Success" if ok else "Failed")
        elif args.forecast:
            embed = client.build_forecast_embed(args.forecast)
            ok = client.post_to_webhook(args.webhook_url, embed)
            logger.info("Posted Forecast to Discord Webhook: %s", "Success" if ok else "Failed")
        else:
            logger.info("No action specified. Use --post-edge, --post-track, or --forecast <q>.")
    else:
        logger.info("Foresea Discord Client initialized. Pass --webhook-url to post.")


if __name__ == "__main__":
    main()
