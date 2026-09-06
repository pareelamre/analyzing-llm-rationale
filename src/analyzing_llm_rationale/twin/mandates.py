"""Immutable owner-approved authority for autonomous, never manual, execution."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from typing import Tuple


@dataclass(frozen=True)
class Mandate:
    id: str
    owner_id: str
    account_scope_id: str
    strategy_version: str
    expires_at: datetime
    live: bool = False
    approved_hash: str | None = None
    revoked: bool = False
    account_epoch: int = 1
    venue: str = "shadow"
    allowed_actions: Tuple[str, ...] = ("BUY_YES", "BUY_NO")
    max_capital: str = "0"
    max_loss: str = "0"
    model_hash: str = "shadow"
    config_hash: str = "shadow"
    readiness_hash: str | None = None

    def digest(self) -> str:
        payload = {
            "id": self.id, "owner_id": self.owner_id, "account_scope_id": self.account_scope_id,
            "strategy_version": self.strategy_version, "expires_at": self.expires_at.isoformat(), "live": self.live,
            "account_epoch": self.account_epoch, "venue": self.venue, "allowed_actions": sorted(self.allowed_actions),
            "max_capital": self.max_capital, "max_loss": self.max_loss, "model_hash": self.model_hash,
            "config_hash": self.config_hash, "readiness_hash": self.readiness_hash,
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def active(self, *, now: datetime) -> bool:
        return not self.revoked and self.approved_hash == self.digest() and now < self.expires_at


def approve(draft: Mandate, *, owner_id: str, readiness_hash: str | None = None) -> Mandate:
    if draft.owner_id != owner_id or draft.revoked:
        raise PermissionError("only the owner can approve an active mandate draft")
    if draft.expires_at.tzinfo is None or draft.account_epoch < 1 or not draft.allowed_actions:
        raise ValueError("mandate scope is incomplete")
    if draft.live and (not readiness_hash or readiness_hash != draft.readiness_hash):
        raise PermissionError("live activation requires the current verified readiness artifact")
    approved = replace(draft, approved_hash=None)
    return replace(approved, approved_hash=approved.digest())


def revoke(mandate: Mandate, *, owner_id: str) -> Mandate:
    if mandate.owner_id != owner_id:
        raise PermissionError("only the owner can revoke a mandate")
    return mandate if mandate.revoked else replace(mandate, revoked=True)