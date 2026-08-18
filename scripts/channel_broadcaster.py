"""Foresea Unified Multi-Channel Broadcaster.

Broadcasts top prediction market edge opportunities, calibrated forecasts,
and market updates across Telegram channels and Discord server webhooks.

Features:
- Reads broadcast targets from a JSON config file (default: data/broadcast_channels.json)
  or environment variables (DISCORD_WEBHOOKS, TELEGRAM_CHANNELS, TELEGRAM_BOT_TOKEN).
- Formats rich HTML messages for Telegram and Discord Embeds for Discord webhooks.
- Configurable minimum edge threshold (--min-edge) and limit (--limit).
- Safe dry-run mode (--dry-run) to preview all broadcast payloads before sending.

Usage:
    # Dry-run preview:
    python scripts/channel_broadcaster.py --dry-run

    # Send broadcasts live to all enabled channels:
    python scripts/channel_broadcaster.py --send --token "$TELEGRAM_BOT_TOKEN"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("foresea-channel-broadcaster")

DEFAULT_FORESEA_URL = "https://foresea.ink"
DEFAULT_CONFIG_PATH = Path("data/broadcast_channels.json")


class ChannelBroadcaster:
    """Dispatches Foresea signals across Telegram and Discord channels."""

    def __init__(
        self,
        foresea_url: str = DEFAULT_FORESEA_URL,
        telegram_token: str = "",
        config_path: Optional[Path] = None,
        dry_run: bool = True,
        session: Optional[requests.Session] = None,
    ):
        self.foresea_url = foresea_url.rstrip("/")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.dry_run = dry_run
        self.session = session or requests.Session()

    def load_targets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load target channels from config file and environment variables."""
        targets: Dict[str, List[Dict[str, Any]]] = {
            "discord_webhooks": [],
            "telegram_channels": [],
        }

        # 1. Load from config file if present
        if self.config_path and self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                for hook in data.get("discord_webhooks", []):
                    if hook.get("url") and hook.get("enabled", True):
                        targets["discord_webhooks"].append(hook)
                for tg in data.get("telegram_channels", []):
                    if tg.get("chat_id") and tg.get("enabled", True):
                        targets["telegram_channels"].append(tg)
            except Exception as exc:
                logger.error("Failed to parse %s: %s", self.config_path, exc)

        # 2. Append from environment variables (comma-separated)
        env_discord = os.getenv("DISCORD_WEBHOOKS", "")
        if env_discord:
            for url in env_discord.split(","):
                u = url.strip()
                if u and not any(h["url"] == u for h in targets["discord_webhooks"]):
                    targets["discord_webhooks"].append({"name": "Env Webhook", "url": u, "enabled": True})

        env_tg = os.getenv("TELEGRAM_CHANNELS", "")
        if env_tg:
            for cid in env_tg.split(","):
                c = cid.strip()
                if c and not any(t["chat_id"] == c for t in targets["telegram_channels"]):
                    targets["telegram_channels"].append({"name": "Env TG Channel", "chat_id": c, "enabled": True})

        return targets

    def fetch_edge_opportunities(self, limit: int = 10, min_edge: float = 0.05, min_credibility: float = 0.60) -> List[Dict[str, Any]]:
        """Fetch top mispriced opportunities from Foresea GET /edge-board and verify credibility."""
        url = f"{self.foresea_url}/edge-board"
        try:
            resp = self.session.get(url, params={"limit": limit * 2}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                raw_list = data.get("opportunities") or data.get("edge_board") if isinstance(data, dict) else (data if isinstance(data, list) else [])
                filtered = []
                for item in raw_list:
                    model_p = item.get("model_probability")
                    mkt_p = item.get("market_probability")
                    edge = item.get("edge") or (abs(model_p - mkt_p) if model_p is not None and mkt_p is not None else 0.0)
                    cred_score = item.get("credibility_score")
                    is_cred = item.get("is_credible", True) if cred_score is None else (cred_score >= min_credibility)

                    if edge >= min_edge and is_cred:
                        filtered.append(item)
                    if len(filtered) >= limit:
                        break
                return filtered
        except Exception as exc:
            logger.error("Failed to fetch edge board from %s: %s", url, exc)
        return []

    def format_telegram_alert(self, opportunities: List[Dict[str, Any]], limit: int = 5) -> str:
        """Format an HTML market edge alert message for Telegram."""
        if not opportunities:
            return ""

        msg = [
            "⚡ <b>FORESEA MARKET EDGE ALERT</b>",
            "<i>Top prediction market mispricings verified by calibrated AI:</i>",
            "",
        ]
        for idx, opp in enumerate(opportunities[:limit], 1):
            title = opp.get("question") or opp.get("title") or "Market"
            platform = opp.get("platform", "Venue")
            mkt_p = opp.get("market_probability")
            model_p = opp.get("model_probability")
            edge = opp.get("edge") or (abs(model_p - mkt_p) if model_p is not None and mkt_p is not None else 0.0)
            rec = opp.get("recommendation") or ("BUY YES" if (model_p or 0) > (mkt_p or 0) else "BUY NO")
            link = opp.get("market_url") or self.foresea_url
            grade = opp.get("credibility_grade") or "A"
            score = opp.get("credibility_score")
            score_str = f"{int(score*100)}%" if score is not None else "Verified"

            mkt_str = f"{mkt_p*100:.0f}%" if mkt_p is not None else "?"
            model_str = f"{model_p*100:.0f}%" if model_p is not None else "?"
            edge_str = f"{edge*100:+.1f}%"

            msg.extend([
                f"<b>{idx}. <a href=\"{link}\">{self._escape_html(title[:65])}</a></b>",
                f"   Venue: <code>{platform}</code> | Action: <b>{rec}</b> | Trust: <code>Grade {grade} ({score_str})</code>",
                f"   Market: <code>{mkt_str}</code> → Foresea: <code>{model_str}</code> (<b>{edge_str} edge</b>)",
                "",
            ])

        msg.append(f'<a href="{self.foresea_url}/#radar">Explore live radar desk on Foresea →</a>')
        return "\n".join(msg)

    def format_discord_embed(self, opportunities: List[Dict[str, Any]], limit: int = 5) -> Dict[str, Any]:
        """Format a rich Discord Embed for Discord webhooks."""
        embed: Dict[str, Any] = {
            "title": "⚡ Foresea Market Edge Alert",
            "description": "Top prediction market mispricings verified by calibrated AI:",
            "color": 0x00D2FF,
            "url": f"{self.foresea_url}/#radar",
            "fields": [],
            "footer": {"text": "Foresea Radar Desk • Credibility Audited • foresea.ink"},
        }

        for idx, opp in enumerate(opportunities[:limit], 1):
            title = opp.get("question") or opp.get("title") or "Market"
            platform = opp.get("platform", "Venue")
            mkt_p = opp.get("market_probability")
            model_p = opp.get("model_probability")
            edge = opp.get("edge") or (abs(model_p - mkt_p) if model_p is not None and mkt_p is not None else 0.0)
            rec = opp.get("recommendation") or ("BUY YES" if (model_p or 0) > (mkt_p or 0) else "BUY NO")
            grade = opp.get("credibility_grade") or "A"

            mkt_str = f"{mkt_p*100:.0f}%" if mkt_p is not None else "?"
            model_str = f"{model_p*100:.0f}%" if model_p is not None else "?"
            edge_str = f"{edge*100:+.1f}%"

            embed["fields"].append({
                "name": f"{idx}. [{platform}] {title[:55]} [Grade {grade}]",
                "value": f"**Action:** `{rec}` | **Market:** `{mkt_str}` → **Foresea:** `{model_str}` (**{edge_str} edge**)",
                "inline": False,
            })

        return embed

    def broadcast_to_telegram(self, chat_id: str, message: str) -> bool:
        """Send message to a Telegram channel/chat."""
        if not self.telegram_token:
            logger.warning("No TELEGRAM_BOT_TOKEN configured. Skipping Telegram broadcast to %s.", chat_id)
            return False

        if self.dry_run:
            logger.info("[DRY RUN TELEGRAM] Would broadcast to %s:\n%s\n", chat_id, message[:200])
            return True

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                logger.info("Successfully broadcast to Telegram channel %s", chat_id)
                return True
            logger.error("Failed Telegram broadcast to %s: HTTP %d %s", chat_id, resp.status_code, resp.text)
        except Exception as exc:
            logger.error("Telegram broadcast error to %s: %s", chat_id, exc)
        return False

    def broadcast_to_discord(self, webhook_url: str, embed: Dict[str, Any], name: str = "") -> bool:
        """Send embed to a Discord Webhook URL."""
        if self.dry_run:
            logger.info("[DRY RUN DISCORD] Would broadcast to %s (%s): %s", name, webhook_url[:35], embed.get("title"))
            return True

        payload = {"embeds": [embed]}
        try:
            resp = self.session.post(webhook_url, json=payload, timeout=10)
            if resp.status_code in (200, 204):
                logger.info("Successfully broadcast to Discord webhook '%s'", name or webhook_url[:35])
                return True
            logger.error("Failed Discord broadcast to '%s': HTTP %d", name, resp.status_code)
        except Exception as exc:
            logger.error("Discord broadcast error to '%s': %s", name, exc)
        return False

    def run_broadcast(self, min_edge: float = 0.05, limit: int = 5) -> Dict[str, int]:
        """Fetch opportunities and broadcast to all configured channels."""
        targets = self.load_targets()
        discord_targets = targets.get("discord_webhooks", [])
        telegram_targets = targets.get("telegram_channels", [])

        total_targets = len(discord_targets) + len(telegram_targets)
        logger.info("Found %d target channels (%d Discord, %d Telegram).", total_targets, len(discord_targets), len(telegram_targets))

        if total_targets == 0:
            logger.info("No active broadcast targets configured. Add channels to %s or set DISCORD_WEBHOOKS / TELEGRAM_CHANNELS.", self.config_path)
            return {"sent": 0, "failed": 0, "skipped": 0}

        opps = self.fetch_edge_opportunities(limit=limit, min_edge=min_edge)
        logger.info("Fetched %d opportunities with edge >= %.1f%%.", len(opps), min_edge * 100)

        if not opps:
            logger.info("No opportunities met the edge threshold of %.1f%%. Skipping broadcast.", min_edge * 100)
            return {"sent": 0, "failed": 0, "skipped": total_targets}

        stats = {"sent": 0, "failed": 0, "skipped": 0}

        # 1. Broadcast to Discord
        if discord_targets:
            discord_embed = self.format_discord_embed(opps, limit=limit)
            for hook in discord_targets:
                url = hook.get("url", "")
                name = hook.get("name", "Discord Webhook")
                if url:
                    ok = self.broadcast_to_discord(url, discord_embed, name=name)
                    if ok:
                        stats["sent"] += 1
                    else:
                        stats["failed"] += 1

        # 2. Broadcast to Telegram
        if telegram_targets:
            tg_message = self.format_telegram_alert(opps, limit=limit)
            for tg in telegram_targets:
                cid = tg.get("chat_id", "")
                if cid:
                    ok = self.broadcast_to_telegram(cid, tg_message)
                    if ok:
                        stats["sent"] += 1
                    else:
                        stats["failed"] += 1

        logger.info("Broadcast run completed: %d sent, %d failed.", stats["sent"], stats["failed"])
        return stats

    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresea Multi-Channel Signal Broadcaster")
    parser.add_argument("--foresea-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL))
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram Bot Token")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to broadcast_channels.json")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Simulate without sending network messages")
    parser.add_argument("--send", action="store_false", dest="dry_run", help="Actually send broadcast messages")
    parser.add_argument("--min-edge", type=float, default=0.06, help="Minimum edge threshold (default: 0.06 = 6%)")
    parser.add_argument("--limit", type=int, default=5, help="Number of opportunities to include")
    args = parser.parse_args()

    broadcaster = ChannelBroadcaster(
        foresea_url=args.foresea_url,
        telegram_token=args.token,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    broadcaster.run_broadcast(min_edge=args.min_edge, limit=args.limit)


if __name__ == "__main__":
    main()
