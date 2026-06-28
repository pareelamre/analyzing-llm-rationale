from __future__ import annotations

import hmac
import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import HTTPException, Request


class RateLimiter:
    """Small fixed-window limiter with Redis support and local fallback."""

    def __init__(self, calls: int, period: int = 60):
        self._calls = calls
        self._period = period
        self._log: dict[str, list[float]] = defaultdict(list)
        self._redis = None
        redis_url = None
        try:
            import os

            redis_url = os.environ.get("REDIS_URL")
        except Exception:
            redis_url = None
        if redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                self._redis.ping()
            except Exception:
                self._redis = None

    def is_allowed(self, key: str) -> bool:
        if self._redis is not None:
            try:
                return self._is_allowed_redis(self._redis, key)
            except Exception:
                pass
        return self._is_allowed_local(key)

    def _is_allowed_redis(self, client: Any, key: str) -> bool:
        bucket = int(time.time() // self._period)
        rkey = f"ratelimit:{key}:{bucket}"
        pipe = client.pipeline()
        pipe.incr(rkey, 1)
        pipe.expire(rkey, self._period)
        count = pipe.execute()[0]
        return int(count) <= self._calls

    def _is_allowed_local(self, key: str) -> bool:
        now = time.monotonic()
        window = now - self._period
        log = self._log[key]
        while log and log[0] < window:
            log.pop(0)
        if len(log) >= self._calls:
            return False
        log.append(now)
        return True


def decode_session(token: str, session_secret: str) -> dict:
    """Verify a session JWT and return its claims."""
    import jwt as _jwt

    try:
        return _jwt.decode(token, session_secret, algorithms=["HS256"])
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired.") from None
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.") from None


def require_session(request: Request, session_secret: str) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer header.")
    return decode_session(auth[7:], session_secret)


def require_auth(
    request: Request,
    required_api_key: Optional[str],
    track_record_token: Optional[str],
    session_secret: str,
) -> dict:
    if required_api_key:
        api_key = request.headers.get("X-API-Key")
        if api_key == required_api_key:
            return {"sub": "api-key-user", "email": "api@foresea.ink", "name": "API Key User"}

    track_token = request.headers.get("X-Track-Token", "")
    if track_record_token and hmac.compare_digest(track_token, track_record_token):
        return {
            "sub": "track-record-action",
            "email": "track-record@foresea.ink",
            "name": "Track Record Action",
        }

    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return decode_session(auth[7:], session_secret)

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Please sign in to use Foresea.",
    )


def optional_predict_claims(
    request: Request,
    required_api_key: Optional[str],
    track_record_token: Optional[str],
    session_secret: str,
) -> Optional[dict]:
    has_auth = bool(request.headers.get("Authorization", "").startswith("Bearer "))
    has_api_key = bool(request.headers.get("X-API-Key"))
    has_track_token = bool(request.headers.get("X-Track-Token"))
    if required_api_key or has_auth or has_api_key or has_track_token:
        return require_auth(request, required_api_key, track_record_token, session_secret)
    return None


def optional_user_id(request: Optional[Request], session_secret: str) -> Optional[str]:
    if request is None:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        return decode_session(auth[7:], session_secret).get("sub")
    except Exception:
        return None


def check_api_key(request: Request, required_api_key: Optional[str]) -> None:
    if not required_api_key:
        return
    if request.headers.get("X-API-Key", "") != required_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def check_rate_limit(
    request: Request,
    limiter: RateLimiter,
    required_api_key: Optional[str],
    track_record_token: Optional[str],
    detail: str,
) -> None:
    if required_api_key and request.headers.get("X-API-Key", "") == required_api_key:
        return
    if track_record_token and hmac.compare_digest(
        request.headers.get("X-Track-Token", ""), track_record_token
    ):
        return
    ip = request.client.host if request.client else "unknown"
    if not limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail=detail,
            headers={"Retry-After": "60"},
        )
