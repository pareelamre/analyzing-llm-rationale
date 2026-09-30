"""Write-ahead cloud audit storage for short-lived Metaculus runners.

The local JSONL is authoritative within a process. A GitHub-hosted runner must
restore it before forecasting and synchronously upload every appended event
before a forecast POST can proceed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

_MAX_AUDIT_BYTES = 64 * 1024 * 1024
_ROTATE_AUDIT_BYTES = 8 * 1024 * 1024
_SAFETY_OUTCOMES = {
    "prepared", "submission_unknown", "submitted_unverified",
    "forecast_verified_comment_pending", "comment_unverified", "submission_verified",
}
_tracer = trace.get_tracer(__name__)
_meter = metrics.get_meter(__name__)
_audit_store_counter = _meter.create_counter("metaculus.audit.remote_operations", unit="1")
_audit_store_duration = _meter.create_histogram("metaculus.audit.remote_duration", unit="s")
logger = logging.getLogger(__name__)


class AuditStorageError(RuntimeError):
    """The remote audit could not be read or updated safely."""


def _client():
    from google.cloud import storage

    return storage.Client()


def _blob(uri: str):
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.path == "/"
        or parsed.query
        or parsed.fragment
        or ".." in parsed.path.split("/")
    ):
        raise AuditStorageError("Invalid Metaculus audit GCS URI.")
    return _client().bucket(parsed.netloc).blob(parsed.path.lstrip("/"))


def _generation_path(path: Path) -> Path:
    return path.with_name(path.name + ".gcs-generation")


def _record(operation: str, started: float, outcome: str) -> None:
    attributes = {"operation": operation, "outcome": outcome}
    _audit_store_counter.add(1, attributes)
    _audit_store_duration.record(perf_counter() - started, attributes)


def restore_audit(path: Path, uri: str) -> None:
    """Restore the latest remote JSONL and its generation before any bot work."""
    started = perf_counter()
    with _tracer.start_as_current_span("metaculus.audit.restore") as span:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = _blob(uri)
            blob.reload(timeout=30)
            if blob.size is not None and blob.size > _MAX_AUDIT_BYTES:
                raise AuditStorageError("Remote Metaculus audit exceeds the size limit.")
            generation = int(blob.generation)
            blob.download_to_filename(str(path), if_generation_match=generation, timeout=30)
            if path.stat().st_size > _MAX_AUDIT_BYTES:
                raise AuditStorageError("Restored Metaculus audit exceeds the size limit.")
            _generation_path(path).write_text(str(generation), encoding="ascii")
            span.set_attribute("payload.bytes", path.stat().st_size)
        except Exception as exc:  # aqg: top-level boundary for remote audit restore
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            _record("restore", started, "failure")
            raise AuditStorageError("Metaculus remote audit restore failed; forecast cycle is disabled.") from exc
        _record("restore", started, "success")
        logger.info("Metaculus remote audit restored")


def upload_audit(path: Path, uri: str) -> None:
    """Conditionally upload the entire JSONL before the caller may publish."""
    started = perf_counter()
    with _tracer.start_as_current_span("metaculus.audit.upload") as span:
        try:
            marker = _generation_path(path)
            raw_generation = marker.read_text(encoding="ascii").strip()
            if not re.fullmatch(r"0|[1-9][0-9]*", raw_generation):
                raise AuditStorageError("Metaculus audit generation marker is invalid.")
            if path.stat().st_size > _MAX_AUDIT_BYTES:
                raise AuditStorageError("Metaculus audit exceeds the size limit.")
            blob = _blob(uri)
            if path.stat().st_size > _ROTATE_AUDIT_BYTES:
                # Archive exact history before compacting the active replay state.
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                archive = _blob(uri + ".archive/" + digest + ".jsonl")
                from google.api_core.exceptions import PreconditionFailed

                try:
                    archive.upload_from_filename(str(path), if_generation_match=0, timeout=30)
                except PreconditionFailed:
                    # Only trust an already-created archive after byte-for-byte readback.
                    if archive.download_as_bytes(timeout=30) != data:
                        raise AuditStorageError("Existing Metaculus audit archive is inconsistent.") from None
                latest: dict[tuple[object, str], str] = {}
                for line in data.decode("utf-8").splitlines():
                    event = json.loads(line)
                    if not isinstance(event, dict) or not isinstance(event.get("question_id"), int):
                        raise AuditStorageError("Metaculus audit event cannot be compacted safely.")
                    kind = "safety" if event.get("outcome") in _SAFETY_OUTCOMES else "other"
                    latest[(event["question_id"], kind)] = line
                path.write_text("\n".join(latest.values()) + "\n", encoding="utf-8")
                logger.warning("Metaculus audit history archived; active replay state compacted")
            blob.upload_from_filename(str(path), if_generation_match=int(raw_generation), timeout=30)
            new_generation = int(blob.generation)
            if new_generation <= int(raw_generation):
                raise AuditStorageError("Metaculus audit upload did not advance its generation.")
            marker.write_text(str(new_generation), encoding="ascii")
            span.set_attribute("payload.bytes", path.stat().st_size)
        except Exception as exc:  # aqg: top-level boundary for write-ahead audit upload
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            _record("upload", started, "failure")
            raise AuditStorageError("Metaculus remote audit upload failed; forecast cycle is disabled.") from exc
        _record("upload", started, "success")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore a Metaculus bot audit before its forecast cycle.")
    parser.add_argument("operation", choices=("restore",))
    parser.add_argument("path", type=Path)
    parser.add_argument("uri")
    args = parser.parse_args(argv)
    from analyzing_llm_rationale.observability import init_observability

    init_observability()
    try:
        restore_audit(args.path, args.uri)
    except AuditStorageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
