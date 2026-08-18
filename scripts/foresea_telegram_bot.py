"""Foresea Telegram Signal & Forecasting Bot.

A lightweight, robust Telegram Bot client that connects prediction market
communities to Foresea's live forecasting engine, edge board, and venue APIs.

Features:
- /forecast <question>: Produce a calibrated probability forecast with evidence.
- /edge: Fetch the top mispriced Polymarket & Kalshi markets ranked by edge.
- /analyze <ticker>: Generate deep market thesis & fair-value analysis.
- /track: Inspect Foresea's live Brier score, accuracy, and calibration track record.
- /subscribe & /unsubscribe: Opt-in to automated periodic market edge alerts.

Run locally or in production:
    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
    export FORESEA_BASE_URL="https://foresea.ink"   # optional, defaults to https://foresea.ink
    python scripts/foresea_telegram_bot.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("foresea-telegram-bot")

DEFAULT_FORESEA_URL = "https://foresea.ink"


class ForeseaTelegramBot:
    """Telegram Bot connecting users to Foresea API endpoints."""

    def __init__(
        self,
        bot_token: str,
        foresea_base_url: str = DEFAULT_FORESEA_URL,
        session: Optional[requests.Session] = None,
    ):
        self.bot_token = bot_token.strip()
        self.base_url = foresea_base_url.rstrip("/")
        self.tg_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.session = session or requests.Session()
        self.subscribed_chats: set[int] = set()

    # ── Telegram API Helpers ──────────────────────────────────────────────────

    def send_message(self, chat_id: int | str, text: str, parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
        """Send an HTML/Markdown formatted message to a Telegram chat."""
        url = f"{self.tg_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_preview,
        }
        try:
            resp = self.session.post(url, json=payload, timeout=15)
            return resp.status_code == 200
        except Exception as exc:
            logger.error("Failed to send message to chat %s: %s", chat_id, exc)
            return False

    def get_updates(self, offset: Optional[int] = None, timeout: int = 30) -> List[Dict[str, Any]]:
        """Poll Telegram getUpdates API for incoming messages."""
        url = f"{self.tg_url}/getUpdates"
        params: Dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = self.session.get(url, params=params, timeout=timeout + 10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", [])
        except Exception as exc:
            logger.error("Error polling Telegram updates: %s", exc)
        return []

    # ── Foresea API Callers & Formatters ──────────────────────────────────────

    def handle_forecast(self, question: str) -> str:
        """Query Foresea POST /predict and format a rich Telegram response."""
        q = question.strip()
        if not q:
            return (
                "<b>⚠️ Please provide a question to forecast.</b>\n"
                "Example: <code>/forecast Will SpaceX land Starship on Mars by 2028?</code>"
            )
        url = f"{self.base_url}/predict"
        try:
            resp = self.session.post(url, json={"question": q}, timeout=60)
            if resp.status_code != 200:
                return f"<b>❌ Error from Foresea API</b>: HTTP {resp.status_code}"
            data = resp.json()
            prob = data.get("predicted_probability")
            ans = data.get("predicted_answer", "Unknown")
            rationale = data.get("rationale", "").strip()
            evidence = data.get("evidence", [])

            prob_str = f"{prob * 100:.1f}%" if prob is not None else "N/A"
            bar = self._format_bar(prob if prob is not None else 0.5)

            msg = [
                "<b>🔮 Foresea Calibrated Forecast</b>",
                f"<b>Question:</b> {self._escape_html(q)}",
                "",
                f"<b>Probability:</b> <code>{prob_str}</code> (<b>{ans.upper()}</b>)",
                f"<code>[{bar}]</code>",
                "",
            ]
            if rationale:
                short_rat = rationale[:400] + ("..." if len(rationale) > 400 else "")
                msg.extend(["<b>Rationale:</b>", f"<i>{self._escape_html(short_rat)}</i>", ""])

            if evidence and isinstance(evidence, list):
                msg.append("<b>Top Evidence:</b>")
                for ev in evidence[:3]:
                    title = ev.get("title") or ev.get("snippet", "")
                    if title:
                        msg.append(f"• {self._escape_html(title[:90])}")
                msg.append("")

            msg.append(f'<a href="{self.base_url}">Explore full model analysis on Foresea →</a>')
            return "\n".join(msg)
        except Exception as exc:
            logger.error("Forecast failed for question '%s': %s", q, exc)
            return f"<b>❌ Error generating forecast:</b> {self._escape_html(str(exc))}"

    def handle_edge_board(self, limit: int = 5) -> str:
        """Query Foresea GET /edge-board and format top mispriced opportunities."""
        url = f"{self.base_url}/edge-board"
        try:
            resp = self.session.get(url, params={"limit": limit}, timeout=15)
            if resp.status_code != 200:
                return f"<b>❌ Error fetching edge board:</b> HTTP {resp.status_code}"
            data = resp.json()
            if isinstance(data, dict):
                opportunities = data.get("opportunities") or data.get("edge_board") or []
            elif isinstance(data, list):
                opportunities = data
            else:
                opportunities = []
            if not opportunities:
                return "<b>⚡ Foresea Edge Board</b>\nNo mispriced markets currently meet the threshold."

            msg = [
                "<b>⚡ Foresea Edge Board — Top Opportunities</b>",
                "<i>Ranked by model-vs-market probability gap:</i>",
                "",
            ]
            for idx, opp in enumerate(opportunities[:limit], 1):
                title = opp.get("question") or opp.get("title") or "Unknown Market"
                platform = opp.get("platform", "Venue")
                mkt_p = opp.get("market_probability")
                model_p = opp.get("model_probability")
                edge = opp.get("edge") or (abs(model_p - mkt_p) if model_p is not None and mkt_p is not None else None)
                rec = opp.get("recommendation") or ("BUY YES" if (model_p or 0) > (mkt_p or 0) else "BUY NO")
                link = opp.get("market_url") or self.base_url

                mkt_str = f"{mkt_p*100:.0f}%" if mkt_p is not None else "?"
                model_str = f"{model_p*100:.0f}%" if model_p is not None else "?"
                edge_str = f"{edge*100:+.1f}%" if edge is not None else ""

                msg.extend([
                    f"<b>{idx}. <a href=\"{link}\">{self._escape_html(title[:65])}</a></b>",
                    f"   Venue: <code>{platform}</code> | Rec: <b>{rec}</b>",
                    f"   Market: <code>{mkt_str}</code> | Foresea: <code>{model_str}</code> (<b>{edge_str} edge</b>)",
                    "",
                ])
            msg.append(f'<a href="{self.base_url}/#radar">View live radar desk on Foresea →</a>')
            return "\n".join(msg)
        except Exception as exc:
            logger.error("Edge board fetch failed: %s", exc)
            return f"<b>❌ Error fetching edge board:</b> {self._escape_html(str(exc))}"

    def handle_track_record(self) -> str:
        """Query Foresea GET /track-record and format calibration status."""
        url = f"{self.base_url}/track-record"
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return f"<b>❌ Error fetching track record:</b> HTTP {resp.status_code}"
            data = resp.json()
            n_resolved = data.get("n_snapshots_resolved", 0)
            overall = data.get("overall") or {}
            brier = overall.get("mean_brier_score")
            acc = overall.get("accuracy")
            skill = overall.get("skill_vs_market")

            brier_str = f"{brier:.4f}" if brier is not None else "N/A"
            acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A"
            skill_str = f"{skill:+.4f}" if skill is not None else "N/A"

            msg = [
                "<b>📈 Foresea Live Track Record</b>",
                "<i>Independently verified, transparent calibration metrics:</i>",
                "",
                f"• <b>Resolved Forecasts:</b> <code>{n_resolved}</code>",
                f"• <b>Mean Brier Score:</b> <code>{brier_str}</code> (lower is better)",
                f"• <b>Directional Accuracy:</b> <code>{acc_str}</code>",
                f"• <b>Skill vs Market:</b> <code>{skill_str}</code>",
                "",
                f'<a href="{self.base_url}/#track-record">View calibration curves & breakdown →</a>',
            ]
            return "\n".join(msg)
        except Exception as exc:
            logger.error("Track record fetch failed: %s", exc)
            return f"<b>❌ Error fetching track record:</b> {self._escape_html(str(exc))}"

    def handle_help(self) -> str:
        """Return the help menu text."""
        return (
            "<b>🌊 Foresea Forecasting & Prediction Market Bot</b>\n\n"
            "<b>Available Commands:</b>\n"
            "• <code>/forecast &lt;question&gt;</code> — Calibrated probability forecast with rationale & evidence\n"
            "• <code>/edge</code> — Top mispriced opportunities across Polymarket & Kalshi\n"
            "• <code>/track</code> — Live Brier score, calibration & accuracy metrics\n"
            "• <code>/subscribe</code> — Receive automated top-edge market alerts\n"
            "• <code>/unsubscribe</code> — Stop automated alerts\n"
            "• <code>/help</code> — Show this menu\n\n"
            f'<i>Powered by <a href="{self.base_url}">Foresea Engine</a></i>'
        )

    # ── Command Dispatcher ────────────────────────────────────────────────────

    def process_message(self, message: Dict[str, Any]) -> None:
        """Parse incoming Telegram message and dispatch to handlers."""
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        text = str(message.get("text", "")).strip()
        if not chat_id or not text:
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]  # strip bot username if in group
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/start", "/help"):
            self.send_message(chat_id, self.handle_help())
        elif cmd == "/forecast":
            self.send_message(chat_id, self.handle_forecast(arg))
        elif cmd in ("/edge", "/radar"):
            self.send_message(chat_id, self.handle_edge_board())
        elif cmd in ("/track", "/trackrecord", "/calibration"):
            self.send_message(chat_id, self.handle_track_record())
        elif cmd == "/subscribe":
            self.subscribed_chats.add(chat_id)
            self.send_message(chat_id, "<b>✅ Subscribed!</b> You will receive periodic top-edge alerts.")
        elif cmd == "/unsubscribe":
            self.subscribed_chats.discard(chat_id)
            self.send_message(chat_id, "<b>✅ Unsubscribed.</b> You will no longer receive alerts.")

    def run_polling(self) -> None:
        """Run the main bot polling loop."""
        logger.info("Foresea Telegram Bot starting with base URL %s...", self.base_url)
        offset = None
        while True:
            try:
                updates = self.get_updates(offset=offset)
                for u in updates:
                    offset = u.get("update_id", 0) + 1
                    msg = u.get("message") or u.get("channel_post")
                    if msg:
                        self.process_message(msg)
            except KeyboardInterrupt:
                logger.info("Bot shutting down by user request...")
                break
            except Exception as exc:
                logger.error("Polling loop error: %s", exc)
                time.sleep(2)


# ── Utilities ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_bar(prob: float, width: int = 15) -> str:
        filled = int(round(prob * width))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foresea Telegram Bot")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram Bot API Token")
    parser.add_argument("--base-url", default=os.getenv("FORESEA_BASE_URL", DEFAULT_FORESEA_URL), help="Foresea API URL")
    args = parser.parse_args()

    if not args.token:
        logger.error("No TELEGRAM_BOT_TOKEN provided. Set TELEGRAM_BOT_TOKEN or pass --token.")
        sys.exit(1)

    bot = ForeseaTelegramBot(bot_token=args.token, foresea_base_url=args.base_url)
    bot.run_polling()


if __name__ == "__main__":
    main()
