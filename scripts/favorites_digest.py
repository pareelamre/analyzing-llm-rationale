#!/usr/bin/env python3
"""Email each user a digest of their favourited markets — runs in a GitHub Action.

This powers the "receive frequent updates" half of the watchlist feature. Users
star markets in the web UI (stored as ``FavoriteMarket`` Datastore entities under
their ``User`` key with ``notify=True``); this job runs on a schedule, fetches the
current market price for each watched market, compares it to the price we last
notified, and sends each user a per-recipient digest of what moved.

It runs off the request path (like ``track_record_tick.py``): a scheduled runner
with Datastore + SMTP credentials, never on Cloud Run.

No-ops cleanly when Datastore or SMTP is not configured, so it is safe to enable
before secrets are wired up.

Env:
  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / ALERT_FROM  — mail transport
  CUSTOM_DOMAIN     public base for links in the email (default https://foresea.ink)
  DIGEST_MIN_MOVE   minimum |price move| (0-1) to include a market (default 0.05)
  DIGEST_FORCE      "1" to email even with no moves (default off)
"""
from __future__ import annotations

import os
import smtplib
import sys
from collections import defaultdict
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import market_data  # noqa: E402

_FAVORITE_KIND = "FavoriteMarket"
_BASE_URL = os.environ.get("CUSTOM_DOMAIN", "https://foresea.ink").rstrip("/")


def _datastore_client():
    try:
        from google.cloud import datastore
        return datastore.Client()
    except Exception as exc:  # missing creds / library
        print(f"datastore unavailable: {exc}")
        return None


def _current_price(platform: str, ident: str) -> Optional[float]:
    platform = (platform or "").lower()
    try:
        if "poly" in platform:
            quote = market_data.fetch_polymarket(slug=ident)
        elif "kalshi" in platform:
            quote = market_data.fetch_kalshi(ident)
        else:
            return None
        prob = quote.get("probability")
        return float(prob) if prob is not None else None
    except Exception:
        return None


def _send(to: str, subject: str, body: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST")
    if not smtp_host or not to:
        return False
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    alert_from = os.environ.get("ALERT_FROM", "noreply@foresea.ink")

    msg = EmailMessage()
    msg["From"] = alert_from
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as s:
            if smtp_user and smtp_password:
                s.starttls()
                s.login(smtp_user, smtp_password)
            s.send_message(msg)
        return True
    except Exception as exc:
        print(f"send failed for {to}: {exc}")
        return False


def _record_event(client: Any, event_name: str, user_id: str, metadata: Dict[str, Any]) -> None:
    try:
        from google.cloud import datastore as _ds
        entity = _ds.Entity(client.key("AnalyticsEvent"), exclude_from_indexes=("metadata", "path"))
        now = datetime.now(timezone.utc)
        entity.update(
            ts=now,
            day=now.strftime("%Y-%m-%d"),
            event_name=event_name,
            path="/digest",
            user_id=user_id,
            visitor_id="digest",
            metadata=str(metadata)[:4000],
        )
        client.put(entity)
    except Exception:
        pass


def _format_line(fav: Dict[str, Any], price: Optional[float], prev: Optional[float]) -> str:
    q = (fav.get("question") or fav.get("key") or "market")[:90]
    if price is None:
        return f"• {q}\n    price unavailable right now"
    now_c = round(price * 100)
    if prev is None:
        return f"• {q}\n    now {now_c}¢  ({fav.get('platform') or ''})"
    move = round((price - prev) * 100)
    arrow = "▲" if move > 0 else "▼" if move < 0 else "→"
    return f"• {q}\n    {arrow} {now_c}¢  ({move:+d}¢ since last update)"


def run() -> int:
    client = _datastore_client()
    if client is None:
        print("No Datastore — nothing to do.")
        return 0
    if not os.environ.get("SMTP_HOST"):
        print("SMTP not configured — nothing to send.")
        return 0

    min_move = float(os.environ.get("DIGEST_MIN_MOVE", "0.05"))
    force = os.environ.get("DIGEST_FORCE", "") in ("1", "true", "yes")

    # All opted-in favourites, grouped by their owning User key.
    query = client.query(kind=_FAVORITE_KIND)
    query.add_filter("notify", "=", True)
    by_user: Dict[Any, List[Any]] = defaultdict(list)
    for entity in query.fetch(limit=5000):
        user_key = entity.key.parent
        if user_key is not None:
            by_user[user_key].append(entity)

    sent = 0
    for user_key, favs in by_user.items():
        user = client.get(user_key)
        email = (user or {}).get("email") if user else None
        if not email:
            continue

        lines: List[str] = []
        updates: List[Any] = []
        for fav in favs:
            price = _current_price(fav.get("platform"), fav.get("ident"))
            prev = fav.get("last_notified_price")
            moved = price is not None and (prev is None or abs(price - prev) >= min_move)
            if moved or force:
                lines.append(_format_line(fav, price, prev))
            if price is not None:
                fav["last_notified_price"] = price
                fav["last_notified_at"] = datetime.now(timezone.utc)
                updates.append(fav)

        if lines:
            body = (
                "Here's what moved on your Foresea watchlist:\n\n"
                + "\n".join(lines)
                + f"\n\nSee them all: {_BASE_URL}/?utm_source=digest#watchlist\n\n"
                "— Foresea\nReply STOP-style: open the app and un-star a market to stop updates."
            )
            if _send(email, "Your Foresea watchlist update", body):
                sent += 1
                _record_event(client, "digest_sent", user_key.name or str(user_key.id), {"markets": len(lines)})
                if updates:
                    client.put_multi(updates)

    print(f"digest complete: emailed {sent} user(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
