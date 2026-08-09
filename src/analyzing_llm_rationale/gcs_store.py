"""GCS-backed sync for the live track record duckdb file.

``data/track_record_store.duckdb`` used to be committed straight to git, but
it outgrew GitHub's 100MB per-file push limit -- every push has been rejected
since, silently dropping new forecast snapshots. CI workflows now download/
upload it from a dedicated GCS bucket via the ``gcloud storage`` CLI; this
module exists only for the live server's two read-only endpoints
(``/market/history``, ``/market/explain-shift``), which have no CLI available
at runtime and need a local copy to open with ``duckdb.connect``.

Mirrors ``server._get_datastore()``'s shape: lazy singleton client,
Application Default Credentials (no explicit project/credentials args).
``ensure_local_copy`` never raises -- every failure degrades to "use whatever
local copy already exists" (or "no copy", if there never was one), matching
how both callers already handle a missing file.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("foresea")

_TRACK_STORE_BUCKET = os.environ.get("TRACK_STORE_BUCKET_NAME", "brave-drive-471109-d9-track-record-store")
_TRACK_STORE_OBJECT = os.environ.get("TRACK_STORE_OBJECT_NAME", "track_record_store.duckdb")

# Re-checking GCS on every single request would put a live network call (and,
# once a new generation lands, a ~90MB download) behind every hit to
# /market/history and /market/explain-shift -- both public and, in history's
# case, unauthenticated-rate-limit-free. A per-process debounce interval caps
# aggregate GCS API volume to "at most 1 check per interval", independent of
# how much request traffic arrives or how many source IPs it's spread across.
_CHECK_INTERVAL_S = float(os.environ.get("TRACK_STORE_CHECK_INTERVAL_S", "60"))
# If nothing has synced successfully in this long, escalate from WARNING to
# ERROR so a permanent break (revoked IAM, renamed object/bucket, expired
# creds) is loud in logs instead of silently serving indefinitely-stale data
# forever behind a per-request WARNING nobody is watching.
_STALE_ALERT_S = float(os.environ.get("TRACK_STORE_STALE_ALERT_S", "3600"))

_gcs_client: Any = None
_lock = Lock()
_last_synced_generation: Optional[int] = None
_last_check_monotonic: Optional[float] = None
_last_success_monotonic: Optional[float] = None
# Staleness baseline when nothing has EVER synced yet -- without this, a
# process that's been failing to sync since the moment it started would read
# as "not stale" forever (stale-since-last-success is undefined, not zero).
_module_loaded_monotonic = time.monotonic()


def _get_gcs_client():
    """Caller must hold ``_lock``."""
    global _gcs_client
    if _gcs_client is None:
        try:
            from google.cloud import storage as _storage
            _gcs_client = _storage.Client()
        except Exception:
            logger.warning("GCS client init failed", exc_info=True)
    return _gcs_client


def ensure_local_copy(local_path: Path) -> bool:
    """Download the track record store from GCS into ``local_path`` if it's
    missing or the bucket has a newer generation than what's already there.
    Skips the GCS round trip entirely if checked within the last
    ``_CHECK_INTERVAL_S`` seconds (success or failure).

    Returns True if ``local_path`` exists and is usable afterward, False if
    there's no local copy and GCS is unavailable/unreachable -- callers
    already handle that the same way they handle a plain missing file. Never
    raises.
    """
    global _last_synced_generation, _last_check_monotonic, _last_success_monotonic
    try:
        with _lock:
            now = time.monotonic()
            if _last_check_monotonic is not None and (now - _last_check_monotonic) < _CHECK_INTERVAL_S:
                return local_path.exists()
            _last_check_monotonic = now

            client = _get_gcs_client()
            if client is None:
                return local_path.exists()

            try:
                blob = client.bucket(_TRACK_STORE_BUCKET).blob(_TRACK_STORE_OBJECT)
                blob.reload()
            except Exception:
                _log_sync_failure("GCS blob metadata check failed")
                return local_path.exists()

            if local_path.exists() and blob.generation == _last_synced_generation:
                _last_success_monotonic = now
                return True

            tmp_path = local_path.with_suffix(local_path.suffix + ".part")
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(tmp_path))
                tmp_path.replace(local_path)
            except Exception:
                _log_sync_failure("GCS download of track record store failed")
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return local_path.exists()

            _last_synced_generation = blob.generation
            _last_success_monotonic = now
            return True
    except Exception:
        # Belt-and-braces: this function must never raise into callers that
        # don't (and shouldn't have to) wrap it in their own try/except.
        logger.warning("ensure_local_copy failed unexpectedly", exc_info=True)
        return local_path.exists()


def _log_sync_failure(message: str) -> None:
    baseline = _last_success_monotonic if _last_success_monotonic is not None else _module_loaded_monotonic
    stale_for = time.monotonic() - baseline
    if stale_for >= _STALE_ALERT_S:
        logger.error("%s (no successful sync in %.0fs)", message, stale_for, exc_info=True)
    else:
        logger.warning(message, exc_info=True)
