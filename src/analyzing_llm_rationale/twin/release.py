"""Fail-closed readiness gate for worker deployment and live activation."""
from __future__ import annotations


def release_ready(*, health: bool, readiness_artifact: bool, unknown_commands: int, owner_authorized: bool) -> bool:
    return health and readiness_artifact and unknown_commands == 0 and owner_authorized
