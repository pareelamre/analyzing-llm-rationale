"""Independent fixed-target trigger for the hosted tournament workflow."""

from __future__ import annotations

import logging
import os
from time import perf_counter

import requests
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

_WORKFLOW = "https://api.github.com/repos/pareelamre/analyzing-llm-rationale/actions/workflows/metaculus-futureeval.yml"
_ACTIVE = ("queued", "in_progress", "waiting", "pending", "requested")
_tracer = trace.get_tracer(__name__)
_meter = metrics.get_meter(__name__)
_counter = _meter.create_counter("metaculus.dispatch.outcomes", unit="1")
_duration = _meter.create_histogram("metaculus.dispatch.duration", unit="s")
logger = logging.getLogger(__name__)


class DispatchError(RuntimeError):
    """Dispatch failed; messages never include response bodies or credentials."""


def dispatch(token: str, session: requests.Session) -> str:
    """Skip visible active work; races are serialized downstream, not locked here."""
    started = perf_counter()
    outcome = "failure"
    with _tracer.start_as_current_span("metaculus.workflow.dispatch") as span:
        try:
            if not token or not token.strip() or any(c.isspace() for c in token):
                raise DispatchError("A nonempty dispatch credential is required.")
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            for status in _ACTIVE:
                response = session.get(
                    f"{_WORKFLOW}/runs", headers=headers,
                    params={"branch": "main", "status": status, "per_page": 1},
                    timeout=10, allow_redirects=False,
                )
                if response.status_code != 200:
                    raise DispatchError("GitHub active-run check failed.")
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
                    raise DispatchError("GitHub returned an invalid run list.")
                if payload["workflow_runs"]:
                    outcome = "skipped"
                    return outcome
            response = session.post(
                f"{_WORKFLOW}/dispatches", headers=headers,
                json={"ref": "main", "inputs": {"submit": "true"}},
                timeout=10, allow_redirects=False,
            )
            if response.status_code != 204:
                raise DispatchError("GitHub did not confirm workflow dispatch.")
            outcome = "dispatched"
            return outcome
        except (requests.RequestException, ValueError, DispatchError):
            # Record only the sanitized error, never the HTTP exception/request.
            error = DispatchError("Tournament workflow dispatch failed.")
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, "DispatchError"))
            raise error from None
        finally:
            span.set_attribute("outcome", outcome)
            _counter.add(1, {"outcome": outcome})
            _duration.record(perf_counter() - started, {"outcome": outcome})
            logger.info("Tournament workflow dispatch outcome=%s", outcome)


def main() -> int:
    from .observability import init_observability

    init_observability()
    logging.basicConfig(level=logging.INFO)
    try:
        with requests.Session() as session:
            session.trust_env = False
            dispatch(os.environ.get("METACULUS_GITHUB_DISPATCH_TOKEN", ""), session)
    except DispatchError:
        logger.error("Tournament workflow dispatch failed; inspect credential and GitHub availability.")
        return 1
    finally:
        for provider in (trace.get_tracer_provider(), metrics.get_meter_provider()):
            flush = getattr(provider, "force_flush", None)
            if callable(flush):
                flush(timeout_millis=5000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
