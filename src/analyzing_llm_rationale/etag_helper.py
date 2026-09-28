"""ETag and HTTP 304 Not Modified evaluation helpers for Foresea polling endpoints.

Provides fast deterministic hash calculation and If-None-Match header matching to
enable external agents and web clients to poll without re-downloading unchanged payloads.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse


def make_etag(data: Any) -> str:
    """Generate a deterministic weak or strong ETag hash from content or a dict/string/bytes."""
    if isinstance(data, (dict, list)):
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, bytes):
        raw = data
    else:
        raw = str(data).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    return f'"{digest}"'


def check_if_none_match(if_none_match_header: Optional[str], etag: str) -> bool:
    """Return True if the client's If-None-Match header matches the computed ETag."""
    if not if_none_match_header:
        return False
    # Normalize weak ETag indicators ('W/') and quotes
    clean_target = etag.strip()
    if clean_target.startswith("W/"):
        clean_target = clean_target[2:]
    clean_target = clean_target.strip('"')

    for part in if_none_match_header.split(","):
        candidate = part.strip()
        if candidate == "*":
            return True
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        candidate = candidate.strip('"')
        if candidate == clean_target:
            return True
    return False


def json_or_304(
    request: Request,
    payload: Any,
    etag: Optional[str] = None,
    cache_control: str = "no-cache, max-age=0, must-revalidate",
) -> Response:
    """Return Response(status_code=304) if If-None-Match matches, otherwise JSONResponse with ETag."""
    if etag is None:
        if isinstance(payload, dict) and payload.get("generated_at"):
            # Include generated_at and top-level item count for fast unique hash
            etag = make_etag(f"{payload.get('generated_at')}:{len(payload)}")
        else:
            etag = make_etag(payload)

    inm = request.headers.get("if-none-match")
    if check_if_none_match(inm, etag):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache_control})

    return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": cache_control})


def bytes_or_304(
    request: Request,
    content: bytes,
    media_type: str = "application/octet-stream",
    etag: Optional[str] = None,
    headers: Optional[dict] = None,
    cache_control: str = "no-cache, max-age=0, must-revalidate",
) -> Response:
    """Return Response(status_code=304) if If-None-Match matches, otherwise binary Response with ETag."""
    if etag is None:
        etag = make_etag(content)
    inm = request.headers.get("if-none-match")
    resp_headers = dict(headers or {})
    resp_headers["ETag"] = etag
    resp_headers["Cache-Control"] = cache_control
    if check_if_none_match(inm, etag):
        return Response(status_code=304, headers=resp_headers)
    return Response(content=content, media_type=media_type, headers=resp_headers)

