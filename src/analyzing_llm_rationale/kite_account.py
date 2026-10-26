"""Read-only Kite Personal account access; no execution or market-data API."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
reads = meter.create_counter("kite.account.reads", unit="1")
duration = meter.create_histogram("kite.account.read.duration", unit="s")

_PATHS = {
    "profile": "/user/profile",
    "holdings": "/portfolio/holdings",
    "positions": "/portfolio/positions",
    "funds": "/user/margins",
    "orders": "/orders",
}
_TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,256}\Z")


class KiteAccountError(ValueError):
    """A safe, user-facing failure; never includes an upstream response body."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def validate_credentials(payload: Any) -> dict[str, str]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"api_key", "access_token"}
        or any(not isinstance(v, str) or not _TOKEN.fullmatch(v) for v in payload.values())
    ):
        raise KiteAccountError("Provide a valid Kite API key and daily access token only.")
    return dict(payload)


def read_account(credentials: dict[str, str], resource: str) -> Any:
    """Read one fixed endpoint, without redirects, retries or response caching."""
    if resource not in _PATHS:
        raise KiteAccountError("Unsupported Kite account resource.")
    credentials = validate_credentials(credentials)
    started = time.perf_counter()
    outcome = "error"
    with tracer.start_as_current_span("kite.account.read", record_exception=False) as span:
        span.set_attribute("kite.resource", resource)
        try:
            try:
                response = requests.get(
                    "https://api.kite.trade" + _PATHS[resource],
                    headers={
                        "X-Kite-Version": "3",
                        "Authorization": "token {api_key}:{access_token}".format(**credentials),
                    },
                    timeout=(3.05, 10),
                    allow_redirects=False,
                )
            except requests.RequestException:
                raise KiteAccountError("Kite is unavailable. Try again later.", 502) from None
            if response.status_code in (401, 403):
                raise KiteAccountError("Kite session expired or was denied. Reconnect with a new daily token.", 409)
            if response.status_code == 429:
                raise KiteAccountError("Kite rate limit reached. Wait before refreshing.", 429)
            if response.status_code != 200:
                raise KiteAccountError("Kite could not return this account resource.", 502)
            try:
                payload = response.json()
            except ValueError:
                raise KiteAccountError("Kite returned an invalid account response.", 502) from None
            expected = list if resource in ("holdings", "orders") else dict
            if (
                not isinstance(payload, dict)
                or payload.get("status") != "success"
                or not isinstance(payload.get("data"), expected)
            ):
                raise KiteAccountError("Kite returned an invalid account response.", 502)
            outcome = "success"
            return payload["data"]
        except KiteAccountError as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.warning("Kite account read failed: resource=%s status=%s", resource, exc.status_code)
            raise
        finally:
            span.set_attribute("outcome", outcome)
            attrs = {"resource": resource, "outcome": outcome}
            reads.add(1, attrs)
            duration.record(time.perf_counter() - started, attrs)
