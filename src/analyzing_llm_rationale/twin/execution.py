"""Authority-checked execution facade; callers inject a shadow or venue adapter."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from .mandates import Mandate


class ExecutionBlocked(RuntimeError):
    pass


def submit_authorized_command(mandate: Mandate, *, now: datetime, live_enabled: bool, submit: Callable[[], object]) -> object:
    if not mandate.active(now=now):
        raise ExecutionBlocked("mandate is inactive")
    if mandate.live and not live_enabled:
        raise ExecutionBlocked("live execution remains disabled")
    return submit()
