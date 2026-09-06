"""Private worker admission checks independent from the public API."""
from __future__ import annotations


class WorkerAuthenticationError(PermissionError):
    pass


def require_worker_request(token: str | None, *, expected_token: str) -> None:
    if not expected_token or token != expected_token:
        raise WorkerAuthenticationError("private worker authentication failed")
