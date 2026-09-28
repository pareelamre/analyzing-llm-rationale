from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebhookSubscription(BaseModel):
    id: str = Field(default_factory=lambda: f"wh_{secrets.token_hex(8)}")
    url: str
    events: List[str] = Field(default_factory=lambda: ["edge_alert", "market_resolution"])
    secret: str = Field(default_factory=lambda: f"whsec_{secrets.token_hex(16)}")
    min_edge: float = 0.08
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    active: bool = True
    user_id: Optional[str] = None


class WebhookManager:
    """Manages active webhook subscriptions and event delivery."""

    def __init__(self) -> None:
        self._subscriptions: Dict[str, WebhookSubscription] = {}

    def register(
        self,
        url: str,
        events: Optional[List[str]] = None,
        min_edge: float = 0.08,
        user_id: Optional[str] = None,
    ) -> WebhookSubscription:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL '{url}'")

        sub = WebhookSubscription(
            url=url,
            events=events or ["edge_alert", "market_resolution"],
            min_edge=min_edge,
            user_id=user_id,
        )
        self._subscriptions[sub.id] = sub
        return sub

    def list_subscriptions(self, user_id: Optional[str] = None) -> List[WebhookSubscription]:
        if user_id:
            return [s for s in self._subscriptions.values() if s.user_id == user_id]
        return list(self._subscriptions.values())

    def get_subscription(self, sub_id: str) -> Optional[WebhookSubscription]:
        return self._subscriptions.get(sub_id)

    def delete_subscription(self, sub_id: str, user_id: Optional[str] = None) -> bool:
        sub = self._subscriptions.get(sub_id)
        if not sub:
            return False
        if user_id and sub.user_id != user_id:
            return False
        del self._subscriptions[sub_id]
        return True

    def dispatch_event(
        self,
        event: str,
        data: Dict[str, Any],
        edge: Optional[float] = None,
    ) -> int:
        """Dispatch event payload to all matching active subscribers asynchronously."""
        dispatched_count = 0
        now_ts = int(time.time())

        for sub in list(self._subscriptions.values()):
            if not sub.active:
                continue
            if "all" not in sub.events and event not in sub.events:
                continue
            if event == "edge_alert" and edge is not None and abs(edge) < sub.min_edge:
                continue

            payload = {
                "event": event,
                "timestamp": now_ts,
                "subscription_id": sub.id,
                "data": data,
            }
            body_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
            signature = generate_signature(sub.secret, body_bytes, timestamp=now_ts)

            headers = {
                "Content-Type": "application/json",
                "X-Foresea-Event": event,
                "X-Foresea-Signature": signature,
                "User-Agent": "Foresea-Webhook-Dispatcher/1.0",
            }

            try:
                # Fire and forget / non-blocking attempt
                requests.post(sub.url, data=body_bytes, headers=headers, timeout=4.0)
                dispatched_count += 1
            except Exception as exc:
                logger.warning("Webhook dispatch failed to %s: %s", sub.url, exc)

        return dispatched_count


def generate_signature(secret: str, payload_bytes: bytes, timestamp: Optional[int] = None) -> str:
    """Generate HMAC-SHA256 signature for webhook payload validation."""
    ts = timestamp or int(time.time())
    signed_payload = f"t={ts}.".encode("utf-8") + payload_bytes
    sig = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


def verify_signature(secret: str, payload_bytes: bytes, header_value: str, tolerance_s: int = 300) -> bool:
    """Verify incoming X-Foresea-Signature header."""
    try:
        parts = dict(item.split("=", 1) for item in header_value.split(","))
        ts_str = parts.get("t")
        expected_sig = parts.get("v1")
        if not ts_str or not expected_sig:
            return False

        ts = int(ts_str)
        if abs(int(time.time()) - ts) > tolerance_s:
            return False

        signed_payload = f"t={ts}.".encode("utf-8") + payload_bytes
        computed = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(computed, expected_sig)
    except Exception:
        return False


_GLOBAL_WEBHOOK_MANAGER = WebhookManager()


def get_webhook_manager() -> WebhookManager:
    return _GLOBAL_WEBHOOK_MANAGER
