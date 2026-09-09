"""Durable, ID-only Cloud Tasks dispatch for the private twin workers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Mapping, Protocol

from google.api_core.exceptions import AlreadyExists
from opentelemetry import metrics, trace

from .worker import WorkerJob, WorkerJobKind, WorkerJobs

tracer = trace.get_tracer(__name__)
dispatch_operations = metrics.get_meter(__name__).create_counter(
    "twin.worker.dispatches", unit="1"
)
duplicate_suppressions = metrics.get_meter(__name__).create_counter(
    "twin.duplicate_suppressions", unit="1"
)
queue_lag_seconds = metrics.get_meter(__name__).create_histogram(
    "twin.queue.lag", unit="s"
)


class WorkerDispatchError(RuntimeError):
    pass


class TaskDispatcher(Protocol):
    def enqueue(self, job: WorkerJob) -> str: ...


@dataclass(frozen=True)
class CloudTasksConfig:
    project_id: str
    location: str
    maintenance_queue: str
    research_queue: str
    maintenance_url: str
    research_url: str
    dispatcher_service_account: str
    maintenance_audience: str
    research_audience: str
    create_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        required = (
            self.project_id, self.location, self.maintenance_queue,
            self.research_queue, self.maintenance_url, self.research_url,
            self.dispatcher_service_account, self.maintenance_audience,
            self.research_audience,
        )
        if any(not str(value).strip() for value in required):
            raise WorkerDispatchError("Cloud Tasks dispatch configuration is incomplete")
        if not self.maintenance_url.startswith("https://") or not self.research_url.startswith("https://"):
            raise WorkerDispatchError("worker targets must use HTTPS")
        if self.create_timeout_seconds <= 0 or self.create_timeout_seconds > 30:
            raise WorkerDispatchError("Cloud Tasks create timeout must be within 30 seconds")


class CloudTasksDispatcher:
    """Create deterministic HTTP tasks whose body contains only a stable job ID."""

    def __init__(self, config: CloudTasksConfig, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            try:
                from google.cloud import tasks_v2
            except ImportError as exc:
                raise WorkerDispatchError("google-cloud-tasks is not installed") from exc
            client = tasks_v2.CloudTasksClient()
            self._post_method = tasks_v2.HttpMethod.POST
        else:
            # HttpMethod.POST is numeric value 1 in the v2 API. A dict request
            # keeps tests independent of the generated protobuf package.
            self._post_method = 1
        self._client = client

    @tracer.start_as_current_span("twin.worker.dispatch")
    def enqueue(self, job: WorkerJob) -> str:
        research = job.kind is WorkerJobKind.RESEARCH
        queue = self.config.research_queue if research else self.config.maintenance_queue
        url = self.config.research_url if research else self.config.maintenance_url
        audience = self.config.research_audience if research else self.config.maintenance_audience
        parent = self._client.queue_path(self.config.project_id, self.config.location, queue)
        task_id = sha256(job.id.encode()).hexdigest()
        task_name = f"{parent}/tasks/{task_id}"
        body = json.dumps({"job_id": job.id}, sort_keys=True, separators=(",", ":")).encode()
        request: Mapping[str, Any] = {
            "parent": parent,
            "task": {
                "name": task_name,
                "http_request": {
                    "http_method": self._post_method,
                    "url": url,
                    "headers": {"Content-Type": "application/json"},
                    "oidc_token": {
                        "service_account_email": self.config.dispatcher_service_account,
                        "audience": audience,
                    },
                    "body": body,
                },
            },
        }
        try:
            response = self._client.create_task(
                request=request, timeout=self.config.create_timeout_seconds,
            )
        except AlreadyExists:
            dispatch_operations.add(1, {"queue_role": "research" if research else "maintenance", "outcome": "duplicate"})
            duplicate_suppressions.add(
                1, {"operation": "task_enqueue", "role": "research" if research else "maintenance"},
            )
            return task_name
        except Exception as exc:
            dispatch_operations.add(1, {"queue_role": "research" if research else "maintenance", "outcome": "error"})
            raise WorkerDispatchError("Cloud Tasks enqueue failed") from exc
        dispatch_operations.add(1, {"queue_role": "research" if research else "maintenance", "outcome": "created"})
        return str(getattr(response, "name", task_name))


def dispatch_due_jobs(
    jobs: WorkerJobs, dispatcher: TaskDispatcher, *, now: datetime, limit: int = 25,
) -> tuple[str, ...]:
    """Enqueue due work in maintenance-first order; durable claims deduplicate delivery."""
    if limit < 1 or limit > 100:
        raise WorkerDispatchError("dispatch limit must be between 1 and 100")
    if now.tzinfo is None or now.utcoffset() is None:
        raise WorkerDispatchError("dispatch needs an aware time")
    task_names: list[str] = []
    for job in jobs.due(now=now)[:limit]:
        if job.created_at is not None:
            queue_lag_seconds.record(
                max(0.0, (now - job.created_at).total_seconds()),
                {"role": "research" if job.kind is WorkerJobKind.RESEARCH else "maintenance",
                 "kind": job.kind.value},
            )
        task_names.append(dispatcher.enqueue(job))
    return tuple(task_names)
