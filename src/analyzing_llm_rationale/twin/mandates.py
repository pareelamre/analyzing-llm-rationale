"""Immutable owner-approved authority for autonomous, never manual, execution."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256


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

    def digest(self) -> str:
        return sha256(f"{self.id}|{self.owner_id}|{self.account_scope_id}|{self.strategy_version}|{self.expires_at.isoformat()}|{self.live}".encode()).hexdigest()

    def active(self, *, now: datetime) -> bool:
        return not self.revoked and self.approved_hash == self.digest() and now < self.expires_at


def approve(draft: Mandate, *, owner_id: str, readiness_hash: str | None = None) -> Mandate:
    if draft.owner_id != owner_id or draft.revoked:
        raise PermissionError("only the owner can approve an active mandate draft")
    if draft.live and not readiness_hash:
        raise PermissionError("live activation requires a verified readiness artifact")
    return replace(draft, approved_hash=draft.digest())


def revoke(mandate: Mandate, *, owner_id: str) -> Mandate:
    if mandate.owner_id != owner_id:
        raise PermissionError("only the owner can revoke a mandate")
    return replace(mandate, revoked=True)
