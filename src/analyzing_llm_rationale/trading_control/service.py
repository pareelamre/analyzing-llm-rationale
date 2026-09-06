"""Small injected execution boundary for confirmed manual orders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol


class CredentialResolver(Protocol):
    """Resolve one user's venue-scoped connection without exposing storage."""

    def __call__(self, user_id: str, venue: str) -> Optional[Mapping[str, Any]]:
        """Return the decrypted connection only for the requested operation."""


class VenueOrderAdapter(Protocol):
    """The narrow submit capability retained by the execution boundary."""

    def __call__(
        self, payload: Mapping[str, Any], *, user_id: str, creds: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Submit a fully confirmed order through one venue adapter."""


@dataclass(frozen=True)
class ConfirmedManualOrderService:
    """Injected connection resolver and venue adapter for manual execution.

    It does not create autonomous authority.  HTTP routes still authenticate,
    apply their current risk checks, and retain their exact confirmation
    behavior before calling this service.
    """

    credential_resolver: CredentialResolver
    venue_adapter: VenueOrderAdapter

    def resolve_credentials(self, *, user_id: str, venue: str) -> Optional[Mapping[str, Any]]:
        return self.credential_resolver(user_id, venue)

    def submit(
        self,
        payload: Mapping[str, Any],
        *,
        user_id: str,
        credentials: Mapping[str, Any],
        confirmation: str,
    ) -> dict[str, Any]:
        return submit_confirmed_manual_order(
            self.venue_adapter,
            payload,
            user_id=user_id,
            credentials=credentials,
            confirmation=confirmation,
        )


def submit_confirmed_manual_order(
    place_order: Callable[..., dict[str, Any]],
    payload: Mapping[str, Any],
    *,
    user_id: str,
    credentials: Mapping[str, Any],
    confirmation: str,
) -> dict[str, Any]:
    """Submit through the supplied adapter after the caller verified authority.

    Callers retain responsibility for authentication, confirmation comparison,
    current guardrails, durable claims, and recovery.  This boundary prevents
    route-specific payload construction from drifting before worker authority
    is introduced in later twin tasks.
    """
    return place_order(
        {**dict(payload), "execute": True, "confirmation": confirmation},
        user_id=user_id,
        creds=credentials,
    )
