from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import ipaddress
import json
import logging
import os
import queue
import random
import re
import secrets
import smtplib
import threading
import time
import traceback
import uuid
from collections import OrderedDict, defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import duckdb
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from analyzing_llm_rationale import agent_capabilities, rag
from analyzing_llm_rationale.pipeline import (
    _parse_json_dict,
    build_user_prompt,
    parse_model_response,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"
_ANALYTICS_DB = Path(os.environ.get("ANALYTICS_DB", "/tmp/foresea_analytics.duckdb"))
_CANONICAL = "https://foresea.ink"
_MCP_ENDPOINT = f"{_CANONICAL}/mcp/"

_REQUIRED_API_KEY: Optional[str] = os.environ.get("API_KEY")
_GOOGLE_CLIENT_ID: Optional[str] = os.environ.get("GOOGLE_CLIENT_ID")
_GITHUB_CLIENT_ID: Optional[str] = os.environ.get("GITHUB_CLIENT_ID")
_GITHUB_CLIENT_SECRET: Optional[str] = os.environ.get("GITHUB_CLIENT_SECRET")
_SESSION_SECRET: str = os.environ.get("SESSION_SECRET", "change-me-in-production")
_SESSION_TTL_DAYS = 30
# The live track record is produced by a GitHub Action and committed to the repo
# (static/track_record_live.json). The server reads that committed file — at
# runtime from raw GitHub (so it tracks hourly commits without a redeploy),
# falling back to the bundled copy, then to the static backtest.
_TRACK_RECORD_LIVE_URL = os.environ.get(
    "TRACK_RECORD_LIVE_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/main/static/track_record_live.json",
)
_TRACK_RECORD_LIVE_TTL = int(os.environ.get("TRACK_RECORD_LIVE_TTL", "600"))
_RADAR_URL = os.environ.get(
    "RADAR_URL",
    "https://raw.githubusercontent.com/pareelamre/analyzing-llm-rationale/main/static/radar.json",
)
_RADAR_TTL = int(os.environ.get("RADAR_TTL", "900"))
_state: Dict[str, Any] = {}
_PUBLIC_MCP = None
_PUBLIC_MCP_APP = None

logger = logging.getLogger("foresea")

# ── Resilience knobs (all free; env-overridable) ──────────────────────────────
# Live forecast calls retry transient upstream failures with bounded backoff and
# a per-attempt timeout, so a flaky SCADS AI response degrades gracefully into a
# clean 503 instead of a raw 500. The batch pipeline already retries; this brings
# the live /predict + agent paths up to the same standard.
_PROVIDER_MAX_RETRIES = int(os.environ.get("PROVIDER_MAX_RETRIES", "2"))      # attempts = retries + 1
_PROVIDER_TIMEOUT_S = float(os.environ.get("PROVIDER_TIMEOUT_S", "90"))       # per-attempt wall-clock budget
_PROVIDER_BACKOFF_BASE_S = float(os.environ.get("PROVIDER_BACKOFF_BASE_S", "0.5"))
# Reject oversized request bodies before they reach a handler. Generous enough
# for PDF uploads on /extract, small enough to blunt memory-exhaustion abuse.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(12 * 1024 * 1024)))
# Flipped to False during graceful shutdown so /ready drains in-flight traffic.
_ready = True

# Conversational system prompt for chat_mode — overrides the JSON-only forecast
# prompt so replies are natural language, not a forecast template.
_CHAT_SYSTEM_PROMPT = (
    "You are Foresea, a helpful forecasting and prediction-market assistant. "
    "Answer the user's question conversationally in clear natural language. "
    "Use light markdown — short paragraphs, **bold** for key points, and bullet or "
    "numbered lists when they help. Ground your answer in the provided evidence "
    "summaries and any prediction-market context when relevant, and be honest about "
    "uncertainty. Treat the provided Current Time as today's date/time for temporal "
    "phrases and deadlines. If the question implies a probability, you may express it in prose "
    "(e.g., \"around 60%\") and, when a market price is given, briefly note whether "
    "you lean above or below it. Do NOT output JSON, key/value objects, or a rigid "
    "forecast template — just talk to the user."
)

_DESCRIPTION = """
## Overview

The **Foresea Intelligence API** turns market questions, linked evidence, and
prediction-market prices into probabilistic forecasts that can sit behind
trading dashboards, alerting systems, and research workflows.

It supports binary, multiple-choice, numeric, and date forecasts. Each response
includes a structured **rationale**, typed forecast fields, optional evidence
sources fetched from live news, and optional market-edge analysis when you pass
a market-implied probability.

---

## Quick start

```bash
curl -X POST https://foresea.ink/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "Will the Fed cut rates before September 30, 2026?",
    "question_type": "binary",
    "market_platform": "Polymarket",
    "market_probability": 0.54,
    "variant": "variant0_neutral_baseline"
  }'
```

---

## Question types

Use `question_type` when your client already knows the shape of the question.
If omitted, the model will try to infer it.

| Type | Response shape |
|---|---|
| `binary` | `predicted_answer` is `Yes`/`No`; `confidence` is 0–1 |
| `multiple_choice` | `options` contains per-option probabilities; `predicted_answer` is the top option |
| `numeric` | `range_forecast` contains `p10`, `p50`, `p90`, and optional `unit` |
| `date` | `range_forecast` contains date percentiles |

For multiple-choice forecasts, pass `options` when the answer set is known.

---

## Prediction-market context

Pass `market_probability` when you know the current market-implied probability
for the target outcome. For binary markets this defaults to the `Yes` side; set
`market_outcome` to compare against `No` or a named multiple-choice outcome.

Foresea returns `market_analysis` with model probability, market probability,
percentage-point edge, and a stance (`model_above_market`, `model_below_market`,
`in_line`, or `not_comparable`).

---

## Trading execution

Foresea can preview and submit guarded orders for Kalshi and Polymarket:

| Endpoint | Purpose |
|---|---|
| `GET /trading/accounts` | Return configured/not-configured venue status without exposing secrets |
| `POST /trading/preview` | Normalize and validate an order without placing it |
| `POST /trading/orders` | Submit a live order after explicit confirmation |

Live trading is disabled unless `FORESEA_ENABLE_TRADING=true`. Every live order
requires a signed-in user session, `execute=true`, and the exact confirmation
phrase `PLACE REAL ORDER`. Market/IOC/FOK-style orders are separately blocked
unless `FORESEA_ALLOW_MARKET_ORDERS=true`. Exchange credentials are read only
from server-side environment variables or Secret Manager mounts.

`/agent/analyze` may return a directional recommendation, but it never places an
order. Use `/trading/preview` first, review the normalized order, then call
`/trading/orders` only when you intend to submit a real order.

---

## Prompt variants

The `variant` field controls how the LLM is prompted. Choose the variant that
best matches the information you have available:

| Variant key | Focus |
|---|---|
| `variant0_neutral_baseline` | Control — no extra framing (default) |
| `variant1_predicted_event` | State the concrete predicted event |
| `variant2_key_attribute` | Highlight time / quantity / actor |
| `variant3_reasoning_type` | Specify reasoning type (speculation, expert forecast…) |
| `variant4_credibility` | Ground rationale in source credibility scores |
| `variant5_key_conditions` | List 2–4 conditions that must hold |
| `variant6_step_by_step_reasoning` | Produce 2–3 numbered reasoning steps |
| `variant7_uncertainty_language` | Require uncertainty hedging words |
| `variant8_temporal_anchors` | Anchor reasoning to specific dates |

---

## Rate limiting

**20 requests per minute per IP address.** Exceeding this returns `429 Too Many Requests`
with a `Retry-After: 60` header.

---

## Authentication

When the server is configured with `API_KEY`, all prediction endpoints require:

```
X-API-Key: <your-key>
```

The `/health` endpoint is always unauthenticated. Per-user endpoints such as
RAG, chat history, and trading require an `Authorization: Bearer <session-token>`
header returned by the sign-in endpoints.

---

## Source code

[github.com/pareelamle/analyzing-llm-rationale](https://github.com/pareelamre/analyzing-llm-rationale)
"""

_TAGS = [
    {
        "name": "Inference",
        "description": "Run forecast and market-edge analysis for binary, multiple-choice, numeric, and date questions.",
    },
    {
        "name": "Markets",
        "description": "Fetch live prices from prediction-market venues (Polymarket, Kalshi).",
    },
    {
        "name": "Trading",
        "description": "Preview and submit guarded prediction-market orders for authenticated users.",
    },
    {
        "name": "Agents",
        "description": "Autonomous analysis: orchestrate market fetch, evidence, forecast, edge, and custom skills into one report.",
    },
    {
        "name": "Knowledge",
        "description": "Per-user RAG knowledge base: ingest documents and retrieve them as evidence.",
    },
    {
        "name": "System",
        "description": "Health and liveness checks.",
    },
]


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _verify_google_token(credential: str) -> dict:
    """Verify a Google One-Tap ID token and return its claims."""
    if not _GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google auth is not configured on this server.")
    try:
        from google.auth.transport.requests import Request as _GRequest
        from google.oauth2.id_token import verify_oauth2_token as _verify
        return _verify(credential, _GRequest(), _GOOGLE_CLIENT_ID)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google credential: {exc}") from exc


def _exchange_github_code(code: str, redirect_uri: Optional[str]) -> dict:
    """Exchange a GitHub OAuth code for a profile (id, email, name, avatar)."""
    if not (_GITHUB_CLIENT_ID and _GITHUB_CLIENT_SECRET):
        raise HTTPException(status_code=503, detail="GitHub auth is not configured on this server.")
    import requests
    try:
        token_resp = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": _GITHUB_CLIENT_ID,
                "client_secret": _GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": redirect_uri or "",
            },
            timeout=12,
        )
        token = token_resp.json().get("access_token")
        if not token:
            raise HTTPException(status_code=401, detail="GitHub did not return an access token.")
        auth_headers = {"Authorization": f"Bearer {token}", "Accept": "application/json",
                        "User-Agent": "foresea-auth"}
        user = requests.get("https://api.github.com/user", headers=auth_headers, timeout=12).json()
        email = user.get("email")
        if not email:
            emails = requests.get("https://api.github.com/user/emails", headers=auth_headers, timeout=12).json()
            if isinstance(emails, list):
                primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
                email = (primary or (emails[0] if emails else {})).get("email", "")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"GitHub sign-in failed: {exc}") from exc
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="GitHub profile could not be read.")
    return {
        "sub": f"github:{user['id']}",
        "email": email or "",
        "name": user.get("name") or user.get("login") or "",
        "picture": user.get("avatar_url") or "",
    }


def _issue_session(sub: str, email: str, name: str, picture: str) -> str:
    """Sign and return a JWT session token."""
    import jwt as _jwt
    now = datetime.now(timezone.utc)
    return _jwt.encode(
        {
            "sub": sub,
            "email": email,
            "name": name,
            "picture": picture,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=_SESSION_TTL_DAYS)).timestamp()),
        },
        _SESSION_SECRET,
        algorithm="HS256",
    )


def _decode_session(token: str) -> dict:
    """Verify a session JWT and return its claims."""
    import jwt as _jwt
    try:
        return _jwt.decode(token, _SESSION_SECRET, algorithms=["HS256"])
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired.") from None
    except _jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token.") from None


def _require_session(request: Request) -> dict:
    """Return session claims from a required bearer token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer header.")
    return _decode_session(auth[7:])


def _optional_user_id(request: Optional[Request]) -> Optional[str]:
    """Decode the bearer token if present and valid; return the user id or None.
    Never raises — used to personalise public endpoints for signed-in users."""
    if request is None:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        return _decode_session(auth[7:]).get("sub")
    except Exception:
        return None


_ds_client: Any = None


def _get_datastore():
    global _ds_client
    if _ds_client is None:
        try:
            from google.cloud import datastore as _ds
            _ds_client = _ds.Client()
        except Exception:
            pass
    return _ds_client


def _upsert_user(sub: str, email: str, name: str, picture: str) -> None:
    """Create or update a User entity in Cloud Datastore."""
    client = _get_datastore()
    if client is None:
        return
    from google.cloud import datastore as _ds
    key = client.key("User", sub)
    entity = client.get(key)
    if entity is None:
        entity = _ds.Entity(key=key, exclude_from_indexes=("picture",))
        entity["created_at"] = datetime.now(timezone.utc)
    entity.update(
        email=email,
        name=name,
        picture=picture,
        last_login=datetime.now(timezone.utc),
    )
    client.put(entity)


# ── Email + password accounts ───────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PBKDF2_ITERATIONS = 200_000
# A throwaway hash so failed logins for unknown emails still pay the PBKDF2 cost,
# keeping response timing uniform whether or not the account exists.
_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$200000$"
    "00000000000000000000000000000000$"
    "0000000000000000000000000000000000000000000000000000000000000000"
)


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def _hash_password(password: str) -> str:
    """Return a PBKDF2-HMAC-SHA256 hash string: ``algo$iters$salt_hex$hash_hex``."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against an encoded PBKDF2 hash."""
    try:
        _algo, iters_str, salt_hex, hash_hex = encoded.split("$")
        iterations = int(iters_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _get_user_record(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored User record (dict) for ``user_id``, or None if absent."""
    client = _get_datastore()
    if client is None:
        return _state.setdefault("password_users", {}).get(user_id)
    entity = client.get(client.key("User", user_id))
    return dict(entity) if entity is not None else None


def _create_password_user(user_id: str, email: str, name: str, password_hash: str) -> None:
    """Persist a new email/password account."""
    now = datetime.now(timezone.utc)
    client = _get_datastore()
    if client is None:
        _state.setdefault("password_users", {})[user_id] = {
            "email": email,
            "name": name,
            "picture": "",
            "password_hash": password_hash,
            "auth_provider": "password",
            "created_at": now,
            "last_login": now,
        }
        return
    from google.cloud import datastore as _ds
    entity = _ds.Entity(
        key=client.key("User", user_id),
        exclude_from_indexes=("picture", "password_hash"),
    )
    entity.update(
        email=email,
        name=name,
        picture="",
        password_hash=password_hash,
        auth_provider="password",
        created_at=now,
        last_login=now,
    )
    client.put(entity)


def _touch_last_login(user_id: str) -> None:
    """Best-effort update of ``last_login`` for an existing account."""
    client = _get_datastore()
    now = datetime.now(timezone.utc)
    if client is None:
        record = _state.setdefault("password_users", {}).get(user_id)
        if record is not None:
            record["last_login"] = now
        return
    entity = client.get(client.key("User", user_id))
    if entity is not None:
        entity["last_login"] = now
        client.put(entity)


def _conversation_key(client: Any, user_id: str, conversation_id: str) -> Any:
    return client.key("User", user_id, "Conversation", conversation_id)


def _message_key(client: Any, user_id: str, conversation_id: str, message_id: str) -> Any:
    return client.key(
        "User",
        user_id,
        "Conversation",
        conversation_id,
        "Message",
        message_id,
    )


def _normalise_message(message: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalised = dict(message)
    normalised.setdefault("id", f"msg_{index}")
    normalised.setdefault("createdAt", normalised.get("updatedAt") or index)
    return normalised


def _list_conversations(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        conversations = list(_state.setdefault("chat_conversations", {}).get(user_id, {}).values())
        return sorted(conversations, key=lambda c: c.get("updatedAt") or 0, reverse=True)
    query = client.query(kind="Conversation", ancestor=client.key("User", user_id))
    conversations = []
    for entity in query.fetch(limit=100):
        conversations.append({
            "id": entity.key.name,
            "title": entity.get("title", "New conversation"),
            "createdAt": entity.get("createdAt"),
            "updatedAt": entity.get("updatedAt"),
            "messages": _list_messages(user_id, entity.key.name),
        })
    conversations.sort(key=lambda c: c.get("updatedAt") or 0, reverse=True)
    return conversations


def _list_messages(user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore()
    if client is None:
        conv = _state.setdefault("chat_conversations", {}).get(user_id, {}).get(conversation_id, {})
        return conv.get("messages", [])
    query = client.query(kind="Message", ancestor=_conversation_key(client, user_id, conversation_id))
    messages = []
    for entity in query.fetch(limit=500):
        message = dict(entity)
        message["id"] = entity.key.name
        messages.append(message)
    messages.sort(key=lambda m: (m.get("createdAt") or 0, m.get("id") or ""))
    return messages


def _put_conversation(user_id: str, conversation: Dict[str, Any]) -> Dict[str, Any]:
    client = _get_datastore()
    messages = [
        _normalise_message(message, index)
        for index, message in enumerate(conversation.get("messages", []))
    ]
    conversation = dict(conversation)
    conversation["messages"] = messages
    if client is None:
        _state.setdefault("chat_conversations", {}).setdefault(user_id, {})[conversation["id"]] = conversation
        return conversation
    from google.cloud import datastore as _ds
    key = _conversation_key(client, user_id, conversation["id"])
    entity = _ds.Entity(key=key)
    entity.update({
        "title": conversation.get("title", "New conversation"),
        "createdAt": conversation.get("createdAt"),
        "updatedAt": conversation.get("updatedAt"),
        "saved_at": datetime.now(timezone.utc),
    })
    message_keys = [
        _message_key(client, user_id, conversation["id"], message["id"])
        for message in messages
    ]
    with client.transaction():
        client.put(entity)
        existing_query = client.query(kind="Message", ancestor=key)
        existing_query.keys_only()
        existing_keys = [message_entity.key for message_entity in existing_query.fetch()]
        stale_keys = [existing_key for existing_key in existing_keys if existing_key not in message_keys]
        if stale_keys:
            client.delete_multi(stale_keys)
        message_entities = []
        for message, message_key in zip(messages, message_keys):
            message_entity = _ds.Entity(key=message_key, exclude_from_indexes=("content", "data"))
            message_entity.update(message)
            message_entities.append(message_entity)
        if message_entities:
            client.put_multi(message_entities)
    return conversation


def _delete_conversation(user_id: str, conversation_id: str) -> None:
    client = _get_datastore()
    if client is None:
        _state.setdefault("chat_conversations", {}).setdefault(user_id, {}).pop(conversation_id, None)
        return
    key = _conversation_key(client, user_id, conversation_id)
    message_query = client.query(kind="Message", ancestor=key)
    message_query.keys_only()
    message_keys = [entity.key for entity in message_query.fetch()]
    if message_keys:
        client.delete_multi(message_keys)
    client.delete(key)


# ── RAG knowledge base (per-user vector store on Datastore) ──────────────────

def _rag_fetch(user_id: str, namespace: str, limit: int = 2000) -> List[Dict[str, Any]]:
    """All stored chunks (with embeddings) for a user/namespace."""
    client = _get_datastore()
    if client is None:
        store = _state.setdefault("rag", {}).get(user_id, [])
        return [c for c in store if c.get("namespace") == namespace]
    query = client.query(kind="VectorChunk", ancestor=client.key("User", user_id))
    query.add_filter("namespace", "=", namespace)
    return [dict(e) for e in query.fetch(limit=limit)]


def _rag_add(user_id: str, namespace: str, items: List[Dict[str, Any]]) -> int:
    """Chunk, embed, and store documents. Returns the number of chunks stored."""
    from analyzing_llm_rationale import rag

    chunks: List[tuple] = []
    for item in items:
        doc_id = item.get("doc_id") or ("doc_" + secrets.token_hex(6))
        for ch in rag.chunk_text(item.get("text") or ""):
            chunks.append((ch, item.get("title") or "", item.get("url") or "", item.get("source") or "", doc_id))
    if not chunks:
        return 0
    vectors = rag.embed([c[0] for c in chunks])
    if not vectors or not vectors[0]:
        raise RuntimeError("Embeddings are unavailable on this server.")
    now = datetime.now(timezone.utc)
    records = [
        {"namespace": namespace, "doc_id": doc_id, "text": text, "title": title,
         "url": url, "source": source, "embedding": emb, "created_at": now}
        for (text, title, url, source, doc_id), emb in zip(chunks, vectors)
    ]
    client = _get_datastore()
    if client is None:
        _state.setdefault("rag", {}).setdefault(user_id, []).extend(records)
        return len(records)
    from google.cloud import datastore as _ds
    entities = []
    for r in records:
        entity = _ds.Entity(
            key=client.key("User", user_id, "VectorChunk"),
            exclude_from_indexes=("text", "embedding", "title", "url", "source"),
        )
        entity.update(r)
        entities.append(entity)
    for i in range(0, len(entities), 100):  # Datastore put_multi caps at 500
        client.put_multi(entities[i:i + 100])
    return len(records)


def _rag_search(user_id: str, namespace: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
    from analyzing_llm_rationale import rag
    qvec = rag.embed_one(query)
    if not qvec:
        return []
    return rag.top_k(qvec, _rag_fetch(user_id, namespace), k=k)


def _rag_documents(user_id: str, namespace: str) -> List[Dict[str, Any]]:
    """Distinct ingested documents (grouped by doc_id) with chunk counts."""
    docs: Dict[str, Dict[str, Any]] = {}
    for c in _rag_fetch(user_id, namespace):
        d = docs.setdefault(c.get("doc_id") or "", {
            "doc_id": c.get("doc_id"), "title": c.get("title"),
            "url": c.get("url"), "source": c.get("source"), "chunks": 0,
        })
        d["chunks"] += 1
    return list(docs.values())


def _rag_delete(user_id: str, namespace: str, doc_id: Optional[str] = None) -> int:
    client = _get_datastore()
    if client is None:
        store = _state.setdefault("rag", {}).get(user_id, [])
        keep = [c for c in store if not (c.get("namespace") == namespace and (doc_id is None or c.get("doc_id") == doc_id))]
        removed = len(store) - len(keep)
        _state["rag"][user_id] = keep
        return removed
    query = client.query(kind="VectorChunk", ancestor=client.key("User", user_id))
    query.add_filter("namespace", "=", namespace)
    keys = [e.key for e in query.fetch() if doc_id is None or e.get("doc_id") == doc_id]
    for i in range(0, len(keys), 200):
        client.delete_multi(keys[i:i + 200])
    return len(keys)


# ── Shared cache / coordination (Redis, optional) ───────────────────────────
_redis_client: Any = None
_redis_initialised = False


def _get_redis() -> Any:
    """Return a shared Redis client when ``REDIS_URL`` is configured, else None.

    Redis (Cloud Memorystore in production) lets multiple Cloud Run instances
    share rate-limit counters and caches. When it is absent or unreachable,
    callers fall back to per-instance in-memory state.
    """
    global _redis_client, _redis_initialised
    if _redis_initialised:
        return _redis_client
    _redis_initialised = True
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    try:
        import redis as _redis  # optional dependency
        client = _redis.from_url(
            url,
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
    return _redis_client


# ── Cache (Redis-backed, per-instance in-memory fallback) ────────────────────
_local_cache: "OrderedDict[str, Any]" = OrderedDict()
_LOCAL_CACHE_MAX = int(os.environ.get("LOCAL_CACHE_MAX", "1024"))
_EVIDENCE_CACHE_TTL = int(os.environ.get("EVIDENCE_CACHE_TTL", "900"))
_EXTRACT_CACHE_TTL = int(os.environ.get("EXTRACT_CACHE_TTL", "3600"))
_PREDICT_CACHE_TTL = int(os.environ.get("PREDICT_CACHE_TTL", "600"))


def _cache_key(*parts: Any) -> str:
    """Stable cache key from arbitrary JSON-serialisable parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Any:
    """Return a cached value (Redis first, then in-memory), or None on miss."""
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(f"cache:{key}")
            return json.loads(raw) if raw is not None else None
        except Exception:
            pass  # fall back to local cache
    entry = _local_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < time.time():
        _local_cache.pop(key, None)
        return None
    _local_cache.move_to_end(key)
    return value


def _cache_set(key: str, value: Any, ttl: int) -> None:
    """Store a JSON-serialisable value with a TTL (seconds). No-op if ttl <= 0."""
    if ttl <= 0:
        return
    client = _get_redis()
    if client is not None:
        try:
            client.set(f"cache:{key}", json.dumps(value, default=str), ex=ttl)
            return
        except Exception:
            pass  # fall back to local cache
    _local_cache[key] = (time.time() + ttl, value)
    _local_cache.move_to_end(key)
    while len(_local_cache) > _LOCAL_CACHE_MAX:
        _local_cache.popitem(last=False)


def _read_live_track_record() -> Optional[Dict[str, Any]]:
    """Return the committed live track-record aggregate, or None.

    Tries (cached): raw GitHub copy → bundled file. Synchronous; call via
    ``run_in_executor`` from async handlers. Fails open to None so the caller
    falls back to the static backtest.
    """
    import requests

    cache_key = _cache_key("track_record_live")
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    payload: Optional[Dict[str, Any]] = None
    try:
        resp = requests.get(_TRACK_RECORD_LIVE_URL, timeout=6)
        if resp.status_code == 200:
            payload = resp.json()
    except Exception:
        logger.warning("live track record fetch failed; trying bundled copy", exc_info=True)
    if payload is None:
        bundled = _STATIC_DIR / "track_record_live.json"
        if bundled.exists():
            try:
                payload = json.loads(bundled.read_text())
            except Exception:
                logger.warning("bundled live track record unreadable", exc_info=True)
    if payload is not None:
        _cache_set(cache_key, payload, _TRACK_RECORD_LIVE_TTL)
    return payload


def _read_radar() -> Optional[Dict[str, Any]]:
    """Return the committed Radar artifact, or None.

    Radar is built by GitHub Actions and committed to ``static/radar.json``.
    Cloud Run only serves the committed JSON (raw GitHub first, bundled file
    second) so venue scraping and batch forecasts never run on visitor requests.
    """
    import requests

    cache_key = _cache_key("radar_artifact")
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    payload: Optional[Dict[str, Any]] = None
    try:
        resp = requests.get(_RADAR_URL, timeout=6)
        if resp.status_code == 200:
            payload = resp.json()
    except Exception:
        logger.warning("radar artifact fetch failed; trying bundled copy", exc_info=True)
    if payload is None:
        bundled = _STATIC_DIR / "radar.json"
        if bundled.exists():
            try:
                payload = json.loads(bundled.read_text())
            except Exception:
                logger.warning("bundled radar artifact unreadable", exc_info=True)
    if payload is not None:
        _cache_set(cache_key, payload, _RADAR_TTL)
    return payload


# ── Rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    """Sliding/fixed-window limiter.

    Uses Redis (shared across instances) when available so the limit holds no
    matter how many Cloud Run instances are running; otherwise it falls back to
    a per-instance in-memory window. The fallback fails open, so a Redis outage
    never blocks traffic — it only loosens enforcement.
    """

    def __init__(self, calls: int = 20, period: int = 60):
        self._calls = calls
        self._period = period
        self._log: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        client = _get_redis()
        if client is not None:
            try:
                return self._is_allowed_redis(client, key)
            except Exception:
                pass  # fall through to in-memory on any Redis error
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


_rate_limiter = _RateLimiter(calls=20, period=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    _ready = True
    logger.info("foresea server starting up")
    async with AsyncExitStack() as stack:
        if _PUBLIC_MCP is not None:
            await stack.enter_async_context(_PUBLIC_MCP.session_manager.run())
            logger.info("foresea public MCP endpoint mounted at /mcp")
        yield
        # Graceful shutdown: flip readiness so the load balancer stops routing new
        # traffic while uvicorn drains in-flight requests on SIGTERM (Cloud Run
        # instance recycle).
        _ready = False
        logger.info("foresea server shutting down (draining in-flight requests)")


app = FastAPI(
    title="Foresea Intelligence API",
    description=_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Pareel Amre",
        "email": "pareel.amre@gmail.com",
        "url": "https://github.com/pareelamre/analyzing-llm-rationale",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://foresea.ink",
        "https://www.foresea.ink",
        "https://REDACTED_CLOUD_RUN_URL.run.app",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "Authorization",
        "Mcp-Session-Id",
        "MCP-Protocol-Version",
    ],
    expose_headers=["Mcp-Session-Id", "X-Request-ID"],
)


def _mount_public_mcp_endpoint() -> None:
    """Expose Foresea as a remote MCP server when the SDK is installed."""
    global _PUBLIC_MCP, _PUBLIC_MCP_APP
    if os.environ.get("DISABLE_PUBLIC_MCP", "").lower() in {"1", "true", "yes"}:
        return
    try:
        from analyzing_llm_rationale.mcp_server import create_mcp_server

        timeout_s = float(os.environ.get("FORESEA_MCP_TIMEOUT_S", "120"))
        _PUBLIC_MCP = create_mcp_server(
            base_url=os.environ.get("FORESEA_MCP_UPSTREAM_URL", _CANONICAL),
            api_key=os.environ.get("FORESEA_MCP_UPSTREAM_API_KEY")
            or os.environ.get("FORESEA_API_KEY")
            or os.environ.get("API_KEY"),
            timeout_s=timeout_s,
            host="0.0.0.0",
            streamable_http_path="/",
        )
        _PUBLIC_MCP_APP = _PUBLIC_MCP.streamable_http_app()
        app.mount("/mcp", _PUBLIC_MCP_APP)
    except Exception as exc:
        _PUBLIC_MCP = None
        _PUBLIC_MCP_APP = None
        logger.warning("public MCP endpoint disabled: %s", exc)


_mount_public_mcp_endpoint()


# Redirect middleware: send requests from run.app hosts to the custom domain
@app.middleware("http")
async def host_redirect_middleware(request: Request, call_next):
    host = (request.headers.get("host") or "").lower()
    # Target domain can be overridden by env var CUSTOM_DOMAIN
    target_domain = os.environ.get("CUSTOM_DOMAIN", "foresea.ink").lower()
    # Redirect only run.app hosts (avoid loop when already on target domain)
    if host.endswith(".run.app") and target_domain and target_domain not in host:
        url = request.url
        new_url = f"https://{target_domain}{url.path}"
        if url.query:
            new_url = new_url + "?" + url.query
        return RedirectResponse(url=new_url, status_code=301)
    return await call_next(request)


# Middleware to catch unhandled exceptions: alert + return a scrubbed response.
# Intentional HTTPExceptions are handled by FastAPI before reaching here, so
# anything caught here is a genuine bug — never leak its traceback to the client.
@app.middleware("http")
async def exception_alert_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        request_id = getattr(request.state, "request_id", "-")
        tb = traceback.format_exc()
        logger.error("unhandled error [%s] %s %s\n%s",
                     request_id, request.method, request.url.path, tb)
        subject = f"Foresea server error: {type(exc).__name__}"
        body = (
            f"Request ID: {request_id}\n"
            f"Request: {request.method} {request.url}\n"
            f"Host: {request.headers.get('host')}\n\n"
            f"Exception:\n{tb}"
        )
        # Send email in background thread to avoid blocking the response.
        try:
            threading.Thread(target=_send_alert_email, args=(subject, body), daemon=True).start()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


# Reject oversized request bodies up front (Content-Length based) so a handler
# never has to buffer a hostile payload.
@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds the {_MAX_BODY_BYTES // (1024 * 1024)} MB limit."},
                )
        except ValueError:
            pass
    return await call_next(request)


# Standard security headers on every response (table-stakes for a public API).
_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    # Permissive enough for the SPA's CDNs (Tailwind, jsDelivr, Google fonts/GIS)
    # while still blocking arbitrary origins, plugins, and framing.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com "
        "https://cdn.jsdelivr.net https://accounts.google.com https://www.gstatic.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net "
        "https://accounts.google.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "connect-src 'self' https://api.openai.com https://openrouter.ai https://accounts.google.com "
        "https://www.gstatic.com; "
        "frame-src https://accounts.google.com; "
        "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    ),
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    for key, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    return response


# Assign a request ID to every request (honouring an inbound X-Request-ID) so
# logs, error responses, and alerts can be correlated. Outermost middleware.
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _send_alert_email(subject: str, body: str) -> None:
    """Send a plain-text alert email. Configured via env vars:
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_FROM, ALERT_TO
    """
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    alert_from = os.environ.get("ALERT_FROM", "noreply@foresea.ink")
    alert_to = os.environ.get("ALERT_TO", "pareel.amre@gmail.com")

    if not smtp_host:
        return

    msg = EmailMessage()
    msg["From"] = alert_from
    msg["To"] = alert_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if smtp_user and smtp_password:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
                s.starttls()
                s.login(smtp_user, smtp_password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as s:
                s.send_message(msg)
    except Exception:
        return


async def _provider_chat(provider, messages, temperature, max_tokens) -> str:
    """Run a blocking ``chat_completion`` in the default executor with a
    per-attempt timeout and bounded exponential backoff (+jitter) on transient
    failures.

    ``ContextLimitError`` and non-retryable ``ProviderResponseError`` propagate
    immediately — only ``RetryableProviderError`` and timeouts are retried.
    Callers map the raised exception to a clean HTTP status via
    :func:`_provider_http_error`.
    """
    from analyzing_llm_rationale.providers import (
        ContextLimitError,
        RetryableProviderError,
    )

    loop = asyncio.get_running_loop()
    attempts = max(1, _PROVIDER_MAX_RETRIES + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: provider.chat_completion(messages, temperature, max_tokens),
                ),
                timeout=_PROVIDER_TIMEOUT_S,
            )
        except ContextLimitError:
            raise
        except (RetryableProviderError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = _PROVIDER_BACKOFF_BASE_S * (2 ** attempt)
            delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
            logger.warning(
                "provider call failed (attempt %d/%d), retrying in %.2fs: %s",
                attempt + 1, attempts, delay, type(exc).__name__,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _provider_stream_chat(provider, messages, temperature, max_tokens):
    """Yield provider tokens from a blocking stream without blocking the event loop."""
    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            for chunk in provider.stream_chat_completion(messages, temperature, max_tokens):
                if chunk:
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield str(item)


def _provider_http_error(exc: Exception) -> HTTPException:
    """Map a provider exception to a clean, non-leaky HTTPException.

    The raw exception text (which can include upstream URLs/keys/prompts) is
    logged server-side, never returned to the client.
    """
    from analyzing_llm_rationale.providers import (
        ContextLimitError,
        RetryableProviderError,
    )

    logger.error("provider error: %s: %s", type(exc).__name__, exc)
    if isinstance(exc, ContextLimitError):
        return HTTPException(
            status_code=422,
            detail="The question plus evidence is too long for the model's context window. Try a shorter question or fewer attachments.",
        )
    if isinstance(exc, (RetryableProviderError, asyncio.TimeoutError)):
        return HTTPException(
            status_code=503,
            detail="The forecasting model is temporarily unavailable. Please retry in a moment.",
            headers={"Retry-After": "10"},
        )
    return HTTPException(status_code=502, detail="The forecasting model returned an unexpected response.")


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    # Revalidate every load so deploys of the single-file SPA show immediately
    # (browser still gets a cheap 304 when unchanged).
    return FileResponse(
        str(_STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache"},
    )


# ── AI-agent / crawler discoverability ────────────────────────────────────────


@app.get("/robots.txt", include_in_schema=False)
async def robots_txt():
    """Welcome search + AI/LLM crawlers (most of these are blocked by default on
    other sites); point them at the sitemap and the machine-readable llms.txt."""
    bots = ["GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot", "anthropic-ai",
            "Claude-Web", "PerplexityBot", "Perplexity-User", "Google-Extended",
            "Applebot-Extended", "CCBot", "Bingbot", "Googlebot"]
    lines = ["# Foresea welcomes AI agents and crawlers.", "User-agent: *", "Allow: /", ""]
    for b in bots:
        lines += [f"User-agent: {b}", "Allow: /", ""]
    lines += [f"Sitemap: {_CANONICAL}/sitemap.xml",
              f"# Machine-readable guide for LLMs: {_CANONICAL}/llms.txt",
              f"# Remote MCP server: {_MCP_ENDPOINT}",
              f"# MCP discovery manifest: {_CANONICAL}/.well-known/mcp/server.json",
              f"# OpenAPI spec: {_CANONICAL}/openapi.json"]
    return PlainTextResponse("\n".join(lines),
                             headers={"Cache-Control": "public, max-age=86400"})


def _mcp_server_manifest() -> Dict[str, Any]:
    return {
        "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        "name": "ink.foresea/forecasting",
        "title": "Foresea Forecasting",
        "description": "Forecast future events and scan prediction-market edges.",
        "version": "1.0.0",
        "websiteUrl": _CANONICAL,
        "repository": {
            "url": "https://github.com/pareelamre/analyzing-llm-rationale",
            "source": "github",
        },
        "remotes": [
            {
                "type": "streamable-http",
                "url": _MCP_ENDPOINT,
            }
        ],
        "_meta": {
            "ink.foresea/tools": [
                "foresea_forecast",
                "foresea_analyze_market",
                "foresea_scan_markets",
                "foresea_track_record",
                "foresea_edge_board",
                "foresea_radar",
            ],
            "ink.foresea/resources": [
                "foresea://track-record",
                "foresea://radar",
                "foresea://openapi.json",
            ],
        },
    }


@app.get("/.well-known/mcp/server.json", include_in_schema=False)
async def mcp_server_json():
    """MCP Registry discovery metadata for Foresea's public remote MCP server."""
    return JSONResponse(_mcp_server_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/.well-known/mcp.json", include_in_schema=False)
async def mcp_json_alias():
    """Compatibility alias for MCP clients that probe the older well-known path."""
    return JSONResponse(_mcp_server_manifest(), headers={"Cache-Control": "public, max-age=86400"})


@app.get("/llms.txt", include_in_schema=False)
async def llms_txt():
    """llms.txt (llmstxt.org): a concise, token-efficient guide that tells an LLM
    agent what Foresea is and exactly how to call it."""
    body = f"""# Foresea

> Foresea turns prediction-market questions into calibrated probability forecasts
> with supporting evidence, a written rationale, and the model-vs-market edge.
> Free to use, with an open JSON API agents can call directly.

## Use the API
- [Remote MCP server]({_MCP_ENDPOINT}): Streamable HTTP MCP endpoint for agents.
  Tools: `foresea_forecast`, `foresea_analyze_market`, `foresea_scan_markets`,
  `foresea_track_record`. Discovery manifest: `{_CANONICAL}/.well-known/mcp/server.json`.
- [Forecast](\
{_CANONICAL}/docs): `POST {_CANONICAL}/predict` with `{{"question": "..."}}` returns a
  structured forecast (binary / multiple-choice / numeric / date), a confidence,
  a rationale, and relevant evidence sources. No auth required.
- [Agent analysis]({_CANONICAL}/docs): `POST {_CANONICAL}/agent/analyze` runs an
  end-to-end analysis of a live market (fetch price, gather evidence, forecast,
  compute edge) and returns one structured report.
- [Edge scan]({_CANONICAL}/docs): `GET {_CANONICAL}/agent/scan?platform=polymarket`
  surfaces mispriced live markets (also `kalshi`, or `all`).
- [Radar]({_CANONICAL}/radar): daily niche-market radar from Reddit, Polymarket,
  Kalshi, and Foresea's committed live snapshots.
- [OpenAPI spec]({_CANONICAL}/openapi.json): full machine-readable API description.

## Track record
- [Live accuracy]({_CANONICAL}/track-record): point-in-time forecasts scored
  against prediction-market outcomes, with skill-vs-market by horizon.

## About
- [Web app]({_CANONICAL}/): ask any forecasting question in natural language.
- [Source](https://github.com/pareelamre/analyzing-llm-rationale)
"""
    return PlainTextResponse(body, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/sitemap.xml", include_in_schema=False)
async def sitemap_xml():
    urls = [(f"{_CANONICAL}/", "daily", "1.0"),
            (f"{_CANONICAL}/track-record", "daily", "0.8"),
            (f"{_CANONICAL}/radar", "daily", "0.8"),
            (f"{_CANONICAL}/docs", "weekly", "0.6")]
    items = "".join(
        f"<url><loc>{loc}</loc><changefreq>{cf}</changefreq><priority>{pr}</priority></url>"
        for loc, cf, pr in urls)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{items}</urlset>'
    return Response(xml, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/track-record/digest", tags=["System"], summary="Shareable track-record summary")
async def track_record_digest():
    """A short, shareable markdown summary of the live track record — ready to
    post (a weekly recap). Built from the resolved-forecast aggregate so it
    never overstates."""
    from analyzing_llm_rationale import track_record_live as trl
    aggregate = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
    return PlainTextResponse(trl.format_digest(aggregate),
                             headers={"Cache-Control": "public, max-age=600"})


class BenchmarkForecast(BaseModel):
    """One resolved forecast to score: a probability and the realized outcome."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Forecast probability of YES (0–1).")
    outcome: int = Field(..., ge=0, le=1, description="Realized outcome: 1 (YES) or 0 (NO).")
    market_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Market price at forecast time, for skill-vs-market.")
    question: Optional[str] = Field(None, max_length=500)


class BenchmarkRequest(BaseModel):
    """Score any forecaster's resolved calls — an AI-forecaster benchmark."""
    forecasts: List[BenchmarkForecast] = Field(..., min_length=1, max_length=5000)
    label: Optional[str] = Field(None, max_length=120, description="Optional name for this forecaster/run.")


@app.post("/benchmark/score", tags=["System"], summary="Score a set of resolved forecasts")
async def benchmark_score(req: BenchmarkRequest, request: Request = None) -> Dict[str, Any]:
    """Grade any forecaster's resolved predictions: Brier score, accuracy, ECE
    (calibration), and — when market prices are supplied — skill vs the market.
    Lets you benchmark an LLM, a person, or another model against the same yardstick
    Foresea grades itself on."""
    if request is not None:
        _check_rate_limit(request)
    from analyzing_llm_rationale import track_record_live as trl
    rows = [{"model_probability": f.probability, "outcome": f.outcome} for f in req.forecasts]
    n = len(rows)
    brier = sum((r["model_probability"] - r["outcome"]) ** 2 for r in rows) / n
    accuracy = sum(1 for r in rows if (r["model_probability"] >= 0.5) == (r["outcome"] == 1)) / n
    ece = trl._ece(rows)
    result: Dict[str, Any] = {
        "label": req.label,
        "n": n,
        "brier_score": round(brier, 4),
        "accuracy": round(accuracy, 4),
        "ece": round(ece, 4) if ece is not None else None,
    }
    market = [f for f in req.forecasts if f.market_probability is not None]
    if len(market) == n:
        market_brier = sum((f.market_probability - f.outcome) ** 2 for f in req.forecasts) / n
        result["market_brier_score"] = round(market_brier, 4)
        result["skill_vs_market"] = round(market_brier - brier, 4)
    return result


@app.get("/track-record", tags=["System"], summary="Public forecasting track record")
async def track_record():
    """Return Foresea's resolved-forecast track record.

    Once the live, point-in-time record has resolved forecasts, this returns it
    (forecasts made on live Polymarket/Kalshi markets and scored at resolution,
    with skill-vs-market). Until then it falls back to the static backtest of
    `gpt-oss-120b` against published Metaculus outcomes: accuracy, Brier score,
    calibration (ECE), a reliability curve, and a sample of resolved forecasts.
    """
    live = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
    if live and live.get("n_snapshots_resolved"):
        return JSONResponse(live, headers={"Cache-Control": "public, max-age=600"})
    path = _STATIC_DIR / "track_record.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Track record not generated yet.")
    return FileResponse(
        str(path),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=600"},
    )


@app.get("/edge-board", tags=["System"], summary="Live model-vs-market edge board")
async def edge_board():
    """Where Foresea's evidence-based fair probability most disagrees with the
    market price right now — and whether that kind of disagreement has paid.

    - ``edge_board``: open markets ranked by live disagreement, each tagged with
      the resolved track record of gaps that size (``track_record.skill_significant``
      flags disagreements our own history has proven beat the market).
    - ``by_edge``: realized skill-vs-market bucketed by disagreement size — the
      proof of whether bigger gaps actually resolved in the model's favour.
    - ``lead_lag``: whether the market has historically moved *toward* the model
      after a disagreement (the model leading the price).
    - ``paper_pnl``: hypothetical paper-trading return of betting the edge over
      resolved snapshots (flat / edge-weighted / validated-only). Paper only — no
      fees or slippage; not live trading.
    """
    live = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
    live = live or {}
    return JSONResponse(
        {
            "generated_at": live.get("generated_at"),
            "edge_board": live.get("edge_board", []),
            "by_edge": live.get("by_edge", []),
            "lead_lag": live.get("lead_lag"),
            "paper_pnl": live.get("paper_pnl"),
            "models_comparison": live.get("models_comparison", []),
            "n_markets_open": live.get("n_markets_open", 0),
            "n_snapshots_resolved": live.get("n_snapshots_resolved", 0),
        },
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/radar", tags=["Markets"], summary="Foresea Radar: niche prediction-market opportunities")
async def radar(limit: int = 10, include_reddit: bool = True):
    """Daily niche-market radar from Reddit, Polymarket, Kalshi, and Foresea's
    committed live snapshots.

    The endpoint serves the committed ``static/radar.json`` artifact built by
    GitHub Actions. The Action may run fresh `/predict` forecasts before
    committing; Cloud Run never does Radar scraping or batch forecasting on page
    load.
    """
    limit = max(1, min(int(limit), 25))
    payload = await asyncio.get_running_loop().run_in_executor(None, _read_radar)
    if payload is None:
        raise HTTPException(status_code=404, detail="Radar artifact not generated yet.")
    payload = dict(payload)
    items = list(payload.get("items") or [])
    payload["items"] = items[:limit]
    if not include_reddit:
        payload["reddit_discussions"] = []
    return JSONResponse(payload, headers={"Cache-Control": f"public, max-age={_RADAR_TTL}"})


# ── Request / response models ─────────────────────────────────────────────────

class NewsArticle(BaseModel):
    """A news article passed as evidence context for the prediction."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Fed signals rate cuts may slow in 2025",
            "source": "Reuters",
            "url": "https://reuters.com/example",
            "publish_date": "2025-01-15T10:30:00Z",
            "summary": "Federal Reserve officials signaled a more cautious approach to rate cuts...",
            "relevance_score": 0.91,
        }
    })

    title: Optional[str] = Field(None, max_length=500, description="Article headline.")
    summary: Optional[str] = Field(None, max_length=4000, description="Short human-written summary.")
    summary_llm: Optional[str] = Field(None, max_length=4000, description="LLM-generated summary.")
    text: Optional[str] = Field(None, max_length=20000, description="Full article body text.")
    source: Optional[str] = Field(None, max_length=200, description="Publisher name (e.g. Reuters).")
    credibility: Optional[Dict[str, Any]] = Field(None, description="Credibility score breakdown.")
    frs: Optional[Dict[str, Any]] = Field(None, description="Future-Resolution Statement metadata.")
    url: Optional[str] = Field(None, max_length=2000, description="Canonical article URL.")
    authors: Optional[Any] = Field(None, description="Author name(s).")
    publish_date: Optional[str] = Field(None, max_length=100, description="ISO 8601 publish timestamp.")
    keywords: Optional[Any] = Field(None, description="Extracted keywords.")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cosine similarity to the question (0–1).")
    search_query: Optional[str] = Field(None, max_length=500, description="Query used to retrieve this article.")


class EvidenceSource(BaseModel):
    """A deduplicated citation drawn from the evidence articles."""

    source: str = Field(..., description="Publisher name.")
    title: Optional[str] = Field(None, description="Article headline.")
    url: Optional[str] = Field(None, description="Article URL.")
    publish_date: Optional[str] = Field(None, description="ISO 8601 publish date.")
    relevance_score: Optional[float] = Field(None, description="Relevance to the question (0–1).")


class PredictRequest(BaseModel):
    """Input payload for a single forecasting prediction."""

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "question": "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
                "question_type": "binary",
                "market_platform": "Polymarket",
                "market_probability": 0.54,
            },
            {
                "question": "Who will win the 2026 Formula 1 drivers championship?",
                "question_type": "multiple_choice",
                "options": ["Max Verstappen", "Lando Norris", "Charles Leclerc", "Lewis Hamilton", "Other"],
            },
            {
                "question": "What will US CPI inflation be in December 2026?",
                "question_type": "numeric",
                "resolution_criteria": "Use the year-over-year CPI-U inflation rate for December 2026.",
                "categories": ["Economics", "United States"],
            },
        ]
    })

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The forecasting question to evaluate. It should ask about a future or "
            "otherwise resolvable event, option, quantity, or date."
        ),
        examples=[
            "Will the Federal Reserve cut interest rates at least once before the end of 2025?",
            "Who will win the 2026 Formula 1 drivers championship?",
            "What will US CPI inflation be in December 2026?",
        ],
    )
    description: str = Field(
        "",
        max_length=4000,
        description="Extended background context that clarifies what the question is asking.",
    )
    resolution_criteria: str = Field(
        "",
        max_length=2000,
        description="Exact conditions or measurement source used to resolve the forecast.",
    )
    categories: List[str] = Field(
        default_factory=list,
        max_length=20,
        description="Topic tags (e.g. `['Economics', 'United States']`).",
    )
    news_articles: List[NewsArticle] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Pre-fetched news articles to use as evidence. "
            "If empty and the evidence pipeline is configured, articles are fetched automatically."
        ),
    )
    variant: str = Field(
        "variant0_neutral_baseline",
        max_length=100,
        description=(
            "Prompt variant that controls the reasoning instruction style. "
            "See the variant table in the Overview section."
        ),
        examples=["variant0_neutral_baseline", "variant3_reasoning_type"],
    )
    attach_evidence: bool = Field(
        True,
        description="If `true` and `news_articles` is empty, fetch live evidence automatically.",
    )
    evidence_top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of evidence articles to retrieve (1–10).",
    )
    created_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 question creation time.")
    publish_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 question publish time.")
    resolve_time: Optional[str] = Field(None, max_length=50, description="ISO 8601 resolution deadline.")
    days_open: Optional[int] = Field(None, ge=0, le=36500, description="Days the question has been open.")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Prior conversation turns for multi-turn context, oldest first. "
            "Each item is `{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}`."
        ),
    )
    question_type: Optional[str] = Field(
        None,
        description=(
            "Question type: `binary`, `multiple_choice`, `numeric`, or `date`. "
            "Set this explicitly for API clients; auto-detected from the question when omitted."
        ),
    )
    options: List[str] = Field(
        default_factory=list,
        max_length=12,
        description="Candidate answers for `multiple_choice` questions (optional; the model can infer them).",
    )
    market_platform: Optional[str] = Field(
        None,
        max_length=80,
        description="Prediction market venue, e.g. `Polymarket`, `Kalshi`, `Manifold`, or `Metaculus`.",
    )
    market_url: Optional[str] = Field(
        None,
        max_length=2000,
        description="URL for the prediction market being analyzed. Must start with `http://` or `https://`.",
    )
    market_outcome: Optional[str] = Field(
        None,
        max_length=120,
        description=(
            "Outcome whose market-implied probability is provided. Defaults to `Yes` for binary markets; "
            "set this to `No` or to a multiple-choice option label when needed."
        ),
    )
    market_probability: Optional[float] = Field(
        None,
        description=(
            "Current market-implied probability for `market_outcome`. Use 0-1, or pass a percentage "
            "from 0-100 and Foresea will normalize it."
        ),
    )
    openrouter_api_key: Optional[str] = Field(
        None,
        max_length=256,
        description=(
            "User-supplied OpenRouter API key. When set, this request is routed through "
            "OpenRouter using `openrouter_model` instead of the server's default provider."
        ),
    )
    openrouter_model: Optional[str] = Field(
        None,
        max_length=128,
        description="OpenRouter model ID (e.g. `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`).",
    )
    model: Optional[str] = Field(
        None,
        max_length=64,
        description=(
            "Optional alternate server-hosted model to forecast with, from the "
            "allowlist (`gpt-oss-120b`, `gemma-4-31b-it`, `kimi-k2.6`). Uses the "
            "server's own key — no BYOK needed. Powers the multi-model paper-trading "
            "comparison."
        ),
    )
    provider_base_url: Optional[str] = Field(
        None,
        max_length=2000,
        description=(
            "Optional OpenAI-compatible chat-completions endpoint (e.g. "
            "`https://api.openai.com/v1/chat/completions`). When set with "
            "`openrouter_api_key` + `openrouter_model`, the request goes to your own "
            "endpoint instead of OpenRouter. Must be public HTTPS."
        ),
    )
    chat_mode: bool = Field(
        False,
        description=(
            "When `true`, skips the forecast output template entirely. "
            "The model responds in plain natural language with no structured JSON."
        ),
    )

    @field_validator("question")
    @classmethod
    def question_must_be_question(cls, v: str) -> str:
        lowered = v.lower()
        injection_signals = [
            "ignore previous", "ignore above", "disregard",
            "system prompt", "you are now", "act as",
            "jailbreak", "do anything now",
        ]
        for signal in injection_signals:
            if signal in lowered:
                raise ValueError("Invalid question content.")
        return v.strip()

    @model_validator(mode="after")
    def _standalone_question_must_be_substantive(self) -> "PredictRequest":
        # A standalone question must be substantive; a follow-up (history present)
        # can be short, e.g. "why?" or "what about June?".
        if not self.history and len((self.question or "").strip()) < 10:
            raise ValueError("question must be at least 10 characters (or include conversation history).")
        return self

    @field_validator("variant")
    @classmethod
    def variant_no_injection(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("Invalid variant name.")
        return v

    @field_validator("question_type")
    @classmethod
    def question_type_supported(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        normalized = v.strip().lower()
        if not normalized:
            return None
        allowed = {"binary", "multiple_choice", "numeric", "date"}
        if normalized not in allowed:
            raise ValueError(f"question_type must be one of {sorted(allowed)}.")
        return normalized

    @field_validator("market_probability", mode="before")
    @classmethod
    def normalize_market_probability(cls, v: Optional[Any]) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise ValueError("market_probability must be a number.") from None
        if 1.0 < value <= 100.0:
            value = value / 100.0
        if value < 0.0 or value > 1.0:
            raise ValueError("market_probability must be between 0 and 1, or 0 and 100 percent.")
        return value

    @field_validator("market_url")
    @classmethod
    def market_url_must_be_http(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        if not value.startswith(("http://", "https://")):
            raise ValueError("market_url must start with http:// or https://.")
        return value

    @field_validator("provider_base_url")
    @classmethod
    def provider_base_url_must_be_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = v.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError("provider_base_url must start with https://.")
        host = (parsed.hostname or "").lower()
        # Block obvious internal targets to limit SSRF against the server's network.
        if (
            not host
            or host == "localhost"
            or host.endswith(".local")
            or host.endswith(".internal")
            or host == "metadata.google.internal"
        ):
            raise ValueError("provider_base_url host is not allowed.")
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None and (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("provider_base_url host is not allowed.")
        return value


class OptionProb(BaseModel):
    """A single option and its probability in a multiple-choice forecast."""

    label: str = Field(..., description="The option text.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability assigned to this option (0–1).")


class RangeForecast(BaseModel):
    """A numeric or date estimate expressed as percentile bounds."""

    p10: Optional[str] = Field(None, description="10th-percentile (low) estimate.")
    p50: Optional[str] = Field(None, description="50th-percentile (median) estimate.")
    p90: Optional[str] = Field(None, description="90th-percentile (high) estimate.")
    unit: Optional[str] = Field(None, description="Unit of the estimate, e.g. 'USD', '%', 'people'.")


class MarketAnalysis(BaseModel):
    """Deterministic comparison between Foresea's forecast and a market price."""

    platform: Optional[str] = Field(None, description="Prediction market venue supplied by the caller.")
    market_url: Optional[str] = Field(None, description="Prediction market URL supplied by the caller.")
    outcome: str = Field(..., description="Outcome being compared against the market price.")
    market_probability: float = Field(..., ge=0.0, le=1.0, description="Market-implied probability for the outcome.")
    model_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Foresea probability for the same outcome.")
    edge: Optional[float] = Field(None, description="Model probability minus market probability.")
    stance: str = Field(..., description="`model_above_market`, `model_below_market`, `in_line`, or `not_comparable`.")
    summary: str = Field(..., description="Short human-readable market comparison.")


class MarketOption(BaseModel):
    """One outcome of a fetched prediction market and its implied probability."""
    label: str = Field(..., description="Outcome label, e.g. 'Yes' or 'No'.")
    probability: Optional[float] = Field(None, description="Market-implied probability (0..1), or null when unpriced.")


class MarketQuote(BaseModel):
    """A live prediction-market quote, normalised for use with `/predict`."""
    platform: str = Field(..., description="Venue: 'Polymarket' or 'Kalshi'.")
    question: str = Field("", description="Market question/title.")
    market_url: str = Field("", description="Canonical market URL.")
    outcome: str = Field("", description="Primary outcome (prefers 'Yes').")
    probability: Optional[float] = Field(None, description="Market-implied probability for `outcome` (0..1).")
    outcomes: List[MarketOption] = Field(default_factory=list, description="All outcomes with their probabilities.")


class TradingAccountStatus(BaseModel):
    """Server-side trading readiness without exposing exchange secrets."""
    trading_enabled: bool
    max_order_notional: float
    allow_market_orders: bool
    confirmation_phrase: str
    credential_source: str
    venues: Dict[str, Dict[str, Any]]


class TradeOrderRequest(BaseModel):
    """A guarded prediction-market order request.

    `/trading/preview` validates this without execution. `/trading/orders` only
    submits when `execute=true`, the confirmation phrase is exact, and server
    live-trading env flags are enabled.
    """
    platform: str = Field(..., max_length=20, description="`kalshi` or `polymarket`.")
    action: str = Field("buy", max_length=10, description="`buy` or `sell`.")
    outcome: str = Field("yes", max_length=10, description="Outcome side: `yes` or `no`.")
    order_type: str = Field("limit", max_length=20, description="`limit` or `market`.")
    price: Optional[float] = Field(
        None,
        gt=0.0,
        lt=1.0,
        description="Limit price, or worst acceptable price for market/IOC-style orders.",
    )
    quantity: float = Field(..., gt=0.0, description="Contracts/shares. For Polymarket market-buy, max_cost may define spend.")
    max_cost: Optional[float] = Field(None, gt=0.0, description="Polymarket market-buy spend cap in USD.")
    ticker: Optional[str] = Field(None, max_length=120, description="Kalshi market ticker.")
    token_id: Optional[str] = Field(None, max_length=200, description="Polymarket CLOB outcome token id.")
    slug: Optional[str] = Field(None, max_length=200, description="Polymarket market slug for token resolution.")
    market_id: Optional[str] = Field(None, max_length=80, description="Polymarket market id for token resolution.")
    time_in_force: Optional[str] = Field(None, max_length=40, description="Venue-specific TIF, e.g. GTC/FOK or good_till_canceled.")
    post_only: bool = Field(False, description="Reject rather than cross the book when supported.")
    reduce_only: bool = Field(False, description="Kalshi reduce-only flag.")
    tick_size: Optional[str] = Field(None, max_length=20, description="Polymarket tick size override.")
    neg_risk: Optional[bool] = Field(None, description="Polymarket negative-risk market flag.")
    client_order_id: Optional[str] = Field(None, max_length=128, description="Caller-provided idempotency/order id.")
    cancel_order_on_pause: bool = Field(False, description="Kalshi cancel-on-pause flag.")
    subaccount: Optional[int] = Field(None, ge=0, le=32, description="Kalshi subaccount number.")
    exchange_index: int = Field(0, ge=0, description="Kalshi exchange shard index; currently 0.")
    execute: bool = Field(False, description="Must be true on `/trading/orders` for live execution.")
    confirmation: Optional[str] = Field(None, max_length=80, description="Must equal the server confirmation phrase.")


class TradeOrderPreviewResponse(BaseModel):
    ok: bool
    platform: str
    would_execute: bool
    requires_confirmation: bool
    confirmation_phrase: str
    trading_enabled: bool
    max_order_notional: float
    estimated_notional: float
    warnings: List[str] = Field(default_factory=list)
    normalized_order: Dict[str, Any]


class TradeOrderResponse(TradeOrderPreviewResponse):
    submitted: bool
    user_id: str
    venue_response: Dict[str, Any]


class AgentSkill(BaseModel):
    """A user-defined analysis step the agent runs over the question + evidence."""
    name: str = Field(..., min_length=1, max_length=60, description="Short skill name, e.g. 'Base rate check'.")
    instruction: str = Field(..., min_length=1, max_length=2000, description="What this skill should analyse.")


class AgentSkillResult(BaseModel):
    name: str
    output: str


class AgentAnalyzeRequest(BaseModel):
    """Ask the analysis agent to work a live question end-to-end."""
    question: Optional[str] = Field(None, max_length=2000, description="Market question. Optional if a market identifier is given.")
    platform: Optional[str] = Field(None, max_length=40, description="'polymarket' or 'kalshi' to fetch a live price.")
    slug: Optional[str] = Field(None, max_length=200, description="Polymarket market slug.")
    market_id: Optional[str] = Field(None, max_length=80, description="Polymarket numeric market id.")
    ticker: Optional[str] = Field(None, max_length=80, description="Kalshi market ticker.")
    market_probability: Optional[float] = Field(None, description="Override market price (0..1 or 0..100) when not fetching.")
    variant: str = Field("variant0_neutral_baseline", max_length=64)
    evidence_top_k: int = Field(5, ge=1, le=10)
    skills: List[AgentSkill] = Field(default_factory=list, max_length=5, description="Up to 5 custom skills to run.")
    builtin_skills: bool = Field(False, description="Also run the built-in forecasting toolkit (base rate, scenario decomposition, red team, key drivers).")
    ground_in_record: bool = Field(False, description="Condition the forecast on the model's own live track-record calibration.")
    tool_loop: bool = Field(False, description="Use a ReAct tool-using loop (model plans + calls tools) instead of the fixed pipeline.")
    max_tool_steps: int = Field(5, ge=1, le=8, description="Max tool calls in the loop.")
    history: List[Dict[str, str]] = Field(default_factory=list, max_length=24, description="Prior conversation turns for follow-up context.")
    openrouter_api_key: Optional[str] = Field(None, max_length=256)
    openrouter_model: Optional[str] = Field(None, max_length=128)
    provider_base_url: Optional[str] = Field(None, max_length=2000)


class AgentReport(BaseModel):
    """End-to-end analysis of a live question produced by the agent."""
    question: str
    pipeline: List[str] = Field(default_factory=list, description="Ordered steps the agent ran.")
    platform: Optional[str] = None
    market_url: Optional[str] = None
    outcome: Optional[str] = None
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge: Optional[float] = None
    stance: Optional[str] = None
    recommendation: str = Field(..., description="`buy_yes`, `buy_no`, `hold`, or `no_market_price`.")
    recommendation_detail: str = ""
    confidence: Optional[float] = None
    question_type: str = "binary"
    thesis: str = ""
    evidence_sources: List["EvidenceSource"] = Field(default_factory=list)
    skills: List[AgentSkillResult] = Field(default_factory=list)
    grounding: Optional[str] = Field(None, description="Track-record self-calibration note applied to the forecast.")
    tool_transcript: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls + observations when the tool loop ran.")


class ScanOpportunity(BaseModel):
    """One mispriced market found by the scan agent."""
    question: str
    market_url: Optional[str] = None
    outcome: str = "Yes"
    market_probability: Optional[float] = None
    model_probability: Optional[float] = None
    edge: Optional[float] = None
    stance: Optional[str] = None
    recommendation: str


class AgentScanResponse(BaseModel):
    """Mispriced markets surfaced by the scan agent, ranked by |edge|."""
    platform: str
    scanned: int = Field(..., description="How many live markets were analysed.")
    opportunities: List[ScanOpportunity] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Prediction result for a single forecasting question."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "question_type": "binary",
            "predicted_answer": "No",
            "confidence": 0.72,
            "options": [],
            "range_forecast": None,
            "rationale": (
                "Current Fed guidance suggests rates will remain elevated through mid-2025. "
                "Recent inflation data remains above the 2% target, reducing the likelihood "
                "of a cut in the near term. However, slowing economic growth introduces some "
                "downside risk that could prompt a cut by year-end."
            ),
            "model_rationale": (
                "Current Fed guidance suggests rates will remain elevated through mid-2025."
            ),
            "variant": "variant0_neutral_baseline",
            "model_key": "gpt-oss-120b",
            "evidence_sources": [
                {
                    "source": "Reuters",
                    "title": "Fed signals rate cuts may slow in 2025",
                    "url": "https://reuters.com/example",
                    "publish_date": "2025-01-15T10:30:00Z",
                    "relevance_score": 0.91,
                }
            ],
            "evidence_articles": [],
            "evidence_error": None,
            "market_analysis": {
                "platform": "Polymarket",
                "market_url": "https://polymarket.com/event/example",
                "outcome": "Yes",
                "market_probability": 0.54,
                "model_probability": 0.72,
                "edge": 0.18,
                "stance": "model_above_market",
                "summary": "Foresea is 18 percentage points above the market on Yes.",
            },
        }
    })

    question_type: str = Field("binary", description="Detected type: `binary`, `multiple_choice`, `numeric`, or `date`.")
    predicted_answer: Optional[str] = Field(None, description="Headline answer: Yes/No, the top option, or the median estimate.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the headline answer (binary/MC). Null for numeric.")
    options: List[OptionProb] = Field(default_factory=list, description="Per-option probabilities for `multiple_choice`.")
    range_forecast: Optional[RangeForecast] = Field(None, description="p10/p50/p90 estimate for `numeric` and `date`.")
    rationale: Optional[str] = Field(None, description="2–4 sentence explanation of the prediction.")
    model_rationale: Optional[str] = Field(None, description="Raw rationale as returned by the model (may differ from `rationale` after post-processing).")
    variant: str = Field(..., description="Prompt variant used for this prediction.")
    model_key: str = Field(..., description="Model identifier (e.g. `gpt-oss-120b`).")
    evidence_sources: List[EvidenceSource] = Field(default_factory=list, description="Deduplicated citations used as evidence.")
    evidence_articles: List[NewsArticle] = Field(default_factory=list, description="Full evidence articles passed to the model.")
    evidence_error: Optional[str] = Field(None, description="Non-null if evidence retrieval failed (prediction still returned).")
    market_analysis: Optional[MarketAnalysis] = Field(None, description="Optional comparison against a supplied prediction-market probability.")


class VertexPredictRequest(BaseModel):
    """Vertex AI prediction contract: wraps one or more `PredictRequest` objects."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "instances": [
                {"question": "Will global oil prices exceed $100 per barrel before the end of 2025?"}
            ]
        }
    })

    instances: List[Dict[str, Any]] = Field(
        ...,
        max_length=10,
        description="Array of `PredictRequest` objects (max 10 per call).",
    )


class VertexPredictResponse(BaseModel):
    """Vertex AI prediction contract: wraps one or more `PredictResponse` objects."""

    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="Array of `PredictResponse` objects, one per input instance.",
    )


class VisitRequest(BaseModel):
    """Anonymous browser visit event."""

    path: str = Field("/", max_length=500)
    referrer: str = Field("", max_length=2000)
    timezone: Optional[str] = Field(None, max_length=100)


class AnalyticsSummary(BaseModel):
    total_visits: int
    unique_visitors: int
    by_day: List[Dict[str, Any]]


class GoogleAuthRequest(BaseModel):
    """Google One-Tap credential submitted by the browser."""
    credential: str = Field(..., max_length=8192)


class GitHubAuthRequest(BaseModel):
    """GitHub OAuth authorization code returned to the browser."""
    code: str = Field(..., max_length=512)
    redirect_uri: Optional[str] = Field(None, max_length=2000)


class RegisterRequest(BaseModel):
    """Email + password sign-up submitted by the browser."""
    email: str = Field(..., max_length=254, description="Account email address.")
    password: str = Field(..., min_length=8, max_length=128, description="At least 8 characters.")
    name: str = Field("", max_length=120, description="Optional display name.")

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        normalised = _normalise_email(v)
        if not _EMAIL_RE.match(normalised):
            raise ValueError("Enter a valid email address.")
        return normalised

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return (v or "").strip()


class LoginRequest(BaseModel):
    """Email + password sign-in submitted by the browser."""
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=128)

    @field_validator("email")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return _normalise_email(v)


class SessionResponse(BaseModel):
    """Issued after a successful Google sign-in."""
    token: str
    user_id: str
    email: str
    name: str
    picture: str


class AuthMeResponse(BaseModel):
    """Current user decoded from a session token."""
    user_id: str
    email: str
    name: str
    picture: str


class ChatConversation(BaseModel):
    """A user-owned chat conversation synced by the browser UI."""
    id: str = Field(..., min_length=1, max_length=120)
    title: str = Field("New conversation", max_length=200)
    createdAt: int = Field(..., ge=0)
    updatedAt: int = Field(..., ge=0)
    messages: List[Dict[str, Any]] = Field(default_factory=list, max_length=200)


class ChatConversationList(BaseModel):
    conversations: List[ChatConversation]


class RagIngestRequest(BaseModel):
    """Add a document to the user's knowledge base."""
    text: Optional[str] = Field(None, max_length=200000, description="Raw text to ingest.")
    url: Optional[str] = Field(None, max_length=2000, description="URL to fetch and ingest.")
    title: str = Field("", max_length=300)
    namespace: str = Field("kb", max_length=40, description="Corpus: 'kb', 'evidence', or 'forecasts'.")


class RagSearchResult(BaseModel):
    text: str = ""
    title: str = ""
    url: str = ""
    source: str = ""
    score: float = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────
def _check_api_key(request: Request) -> None:
    if not _REQUIRED_API_KEY:
        return
    if request.headers.get("X-API-Key", "") != _REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — 20 requests per minute per IP.",
            headers={"Retry-After": "60"},
        )


def _clean_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _clean_article(article: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(article)
    for field in ("title", "summary", "summary_llm", "text", "source"):
        raw = cleaned.get(field)
        cleaned[field] = _clean_text(raw)
        if isinstance(cleaned[field], str) and field == "text":
            cleaned[field] = cleaned[field][:20000]
    return cleaned


def _news_articles(articles: List[Dict[str, Any]]) -> List["NewsArticle"]:
    """Build NewsArticle models, skipping/repairing any that fail validation so
    one malformed source never 500s the whole response."""
    out: List["NewsArticle"] = []
    for a in articles:
        try:
            out.append(NewsArticle(**a))
        except Exception:
            try:
                out.append(NewsArticle(
                    title=(str(a.get("title") or "")[:500]) or None,
                    summary=(str(a.get("summary") or "")[:4000]) or None,
                    source=(str(a.get("source") or "")[:200]) or None,
                    url=(str(a.get("url") or "")[:2000]) or None,
                ))
            except Exception:
                continue
    return out


def _evidence_sources(articles: List[Dict[str, Any]]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen: set = set()
    for article in articles:
        source = (article.get("source") or "Unknown source").strip()
        url = article.get("url") or ""
        key = (source, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(EvidenceSource(
            source=source,
            title=article.get("title"),
            url=article.get("url"),
            publish_date=article.get("publish_date"),
            relevance_score=article.get("relevance_score"),
        ))
    return sources


# ── Multi-type forecasting ────────────────────────────────────────────────────
_TYPE_SCHEMAS = {
    "binary": (
        '{"type":"binary","predicted_answer":"Yes"|"No",'
        '"confidence":0-1,"rationale":"..."}'
    ),
    "multiple_choice": (
        '{"type":"multiple_choice","options":[{"label":"...","probability":0-1}],'
        '"rationale":"..."}'
    ),
    "numeric": (
        '{"type":"numeric","p10":<low>,"p50":<median>,"p90":<high>,'
        '"unit":"...","rationale":"..."}'
    ),
    "date": (
        '{"type":"date","p10":"YYYY-MM-DD","p50":"YYYY-MM-DD",'
        '"p90":"YYYY-MM-DD","rationale":"..."}'
    ),
}


def _typing_instruction(
    question_type: Optional[str],
    options: List[str],
    has_history: bool = False,
) -> str:
    schemas = (
        f"- binary: {_TYPE_SCHEMAS['binary']}\n"
        f"- multiple_choice: {_TYPE_SCHEMAS['multiple_choice']} (probabilities sum to about 1)\n"
        f"- numeric: {_TYPE_SCHEMAS['numeric']}\n"
        f"- date: {_TYPE_SCHEMAS['date']}"
    )
    if has_history:
        # Conversational mode: allow plain text for follow-ups
        instr = (
            "\n\nIf this message is a follow-up, clarification, or discussion about a prior forecast, "
            "respond in plain natural language — no JSON.\n"
            "If it is a new forecasting question, respond with ONLY one JSON object:\n" + schemas
        )
    else:
        instr = (
            "\n\nForecast output contract: choose the schema that matches the question type. "
            "This contract overrides any earlier variant template that says the answer must be Yes or No. "
            "Only binary questions should use a Yes/No `predicted_answer`. "
            "Respond with ONLY one JSON object (no prose).\n"
            "First infer the question type, then use the matching schema:\n" + schemas
        )
    if question_type:
        instr += f"\nThe question type is '{question_type}'. Use that schema."
    if options:
        joined = ", ".join(str(o) for o in options[:12])
        instr += f"\nFor multiple_choice, assign probabilities across: {joined}."
    if not has_history:
        instr += "\nUse `confidence` only for binary forecasts; for multiple_choice, use option probabilities."
    instr += (
        "\nBefore writing the JSON, internally use this forecasting checklist: "
        "rephrase the resolution target, consider arguments for and against, "
        "weigh the most important drivers, check relevant base rates, and adjust "
        "for overconfidence. Do not reveal the checklist; only return the requested JSON."
    )
    return instr


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_percentage_points(value: float) -> str:
    return f"{abs(value) * 100:.0f}"


def _model_probability_for_market(
    question_type: str,
    predicted_answer: Optional[str],
    confidence: Optional[float],
    options: List[OptionProb],
    outcome: str,
) -> Optional[float]:
    target = outcome.strip().lower()
    predicted = (predicted_answer or "").strip().lower()

    if question_type == "binary":
        if confidence is None:
            return None
        if target in ("yes", "y"):
            if predicted == "yes":
                return confidence
            if predicted == "no":
                return 1.0 - confidence
        if target in ("no", "n"):
            if predicted == "no":
                return confidence
            if predicted == "yes":
                return 1.0 - confidence
        return None

    if question_type == "multiple_choice":
        for option in options:
            if option.label.strip().lower() == target:
                return option.probability
        return None

    return None


def _build_market_analysis(
    req: "PredictRequest",
    question_type: str,
    predicted_answer: Optional[str],
    confidence: Optional[float],
    options: List[OptionProb],
) -> Optional[MarketAnalysis]:
    if req.market_probability is None:
        return None

    outcome = (req.market_outcome or ("Yes" if question_type == "binary" else predicted_answer) or "").strip()
    if not outcome:
        outcome = "selected outcome"

    model_probability = _model_probability_for_market(
        question_type=question_type,
        predicted_answer=predicted_answer,
        confidence=confidence,
        options=options,
        outcome=outcome,
    )

    if model_probability is None:
        return MarketAnalysis(
            platform=req.market_platform,
            market_url=req.market_url,
            outcome=outcome,
            market_probability=req.market_probability,
            model_probability=None,
            edge=None,
            stance="not_comparable",
            summary=(
                "Foresea needs a binary or matching multiple-choice forecast "
                "to compare this market price."
            ),
        )

    edge = model_probability - req.market_probability
    if abs(edge) < 0.03:
        stance = "in_line"
        summary = f"Foresea is within 3 percentage points of the market on {outcome}."
    elif edge > 0:
        stance = "model_above_market"
        summary = (
            f"Foresea is {_format_percentage_points(edge)} percentage points "
            f"above the market on {outcome}."
        )
    else:
        stance = "model_below_market"
        summary = (
            f"Foresea is {_format_percentage_points(edge)} percentage points "
            f"below the market on {outcome}."
        )

    return MarketAnalysis(
        platform=req.market_platform,
        market_url=req.market_url,
        outcome=outcome,
        market_probability=req.market_probability,
        model_probability=model_probability,
        edge=edge,
        stance=stance,
        summary=summary,
    )


def _build_typed_response(
    req: "PredictRequest",
    parsed: Optional[Dict[str, Any]],
    content: str,
    evidence_articles: List[Dict[str, Any]],
    evidence_error: Optional[str],
) -> "PredictResponse":
    qtype = (req.question_type or (parsed.get("type") if parsed else None) or "binary").lower()
    rationale = parsed.get("rationale") if parsed else None
    model_key = _model_key_for_request(req)
    base = dict(
        variant=req.variant,
        model_key=model_key,
        evidence_sources=_evidence_sources(evidence_articles),
        evidence_articles=_news_articles(evidence_articles),
        evidence_error=evidence_error,
    )

    if qtype == "multiple_choice" and parsed:
        opts: List[OptionProb] = []
        for o in parsed.get("options") or []:
            if isinstance(o, dict) and o.get("label") is not None:
                try:
                    p = float(o.get("probability"))
                except (TypeError, ValueError):
                    p = 0.0
                opts.append(OptionProb(label=str(o["label"]), probability=max(0.0, min(1.0, p))))
        top = max(opts, key=lambda x: x.probability) if opts else None
        predicted_answer = top.label if top else None
        confidence = top.probability if top else None
        return PredictResponse(
            question_type="multiple_choice",
            options=opts,
            predicted_answer=predicted_answer,
            confidence=confidence,
            rationale=rationale,
            model_rationale=rationale,
            market_analysis=_build_market_analysis(
                req, "multiple_choice", predicted_answer, confidence, opts
            ),
            **base,
        )

    if qtype in ("numeric", "date") and parsed:
        rf = RangeForecast(
            p10=_to_str(parsed.get("p10")),
            p50=_to_str(parsed.get("p50")),
            p90=_to_str(parsed.get("p90")),
            unit=_to_str(parsed.get("unit")),
        )
        return PredictResponse(
            question_type=qtype,
            range_forecast=rf,
            predicted_answer=_to_str(parsed.get("p50")),
            confidence=None,
            rationale=rationale,
            model_rationale=rationale,
            market_analysis=_build_market_analysis(
                req, qtype, _to_str(parsed.get("p50")), None, []
            ),
            **base,
        )

    # binary (default) — reuse the battle-tested parser
    bparsed = parse_model_response(content, ("predicted_answer", "confidence", "rationale"))
    # If no structured forecast fields were found and the response doesn't look like JSON,
    # treat it as a plain conversational reply.
    if not bparsed.get("predicted_answer") and not content.strip().startswith("{"):
        text = content.strip()
        return PredictResponse(
            question_type="chat",
            rationale=text, model_rationale=text, **base,
        )
    brat = bparsed.get("rationale")
    predicted_answer = bparsed.get("predicted_answer")
    confidence = bparsed.get("confidence")
    return PredictResponse(
        question_type="binary",
        predicted_answer=predicted_answer,
        confidence=confidence,
        rationale=brat,
        model_rationale=brat,
        market_analysis=_build_market_analysis(
            req, "binary", predicted_answer, confidence, []
        ),
        **base,
    )


def _model_key_for_request(req: "PredictRequest") -> str:
    model_label = (req.model or "").strip()
    if model_label and model_label in _SCADS_MODEL_ALLOWLIST:
        return model_label
    return req.openrouter_model or _state["model_key"]


async def _prepare_predict_messages(
    req: "PredictRequest",
    rag_user_id: Optional[str],
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[str]]:
    prompt_text = _state["prompt_templates"][req.variant]
    system_prompt = _state["system_prompt"]

    record = req.model_dump()
    evidence_articles = [article.model_dump() for article in req.news_articles]
    evidence_error = None

    # A short follow-up in an ongoing thread ("WE is 90+", "why?") makes a poor
    # search query and derails on literal matches, so answer it from the
    # conversation instead of fetching fresh evidence. Substantive questions
    # (even mid-thread) still retrieve.
    short_followup = bool(req.history) and len(req.question.split()) <= 6

    if req.attach_evidence and not evidence_articles and not short_followup:
        evidence_pipeline = _state.get("evidence_pipeline")
        if evidence_pipeline is None:
            evidence_error = "Evidence pipeline is not configured on this server."
        else:
            top_k = max(1, min(req.evidence_top_k, 10))
            loop = asyncio.get_running_loop()
            evidence_cache_key = _cache_key("evidence", req.question, top_k)
            evidence_articles = _cache_get(evidence_cache_key)
            if evidence_articles is None:
                try:
                    evidence_articles = await loop.run_in_executor(
                        None,
                        lambda: evidence_pipeline.fetch_summarize_rank(req.question, top_k=top_k),
                    )
                    _cache_set(evidence_cache_key, evidence_articles, _EVIDENCE_CACHE_TTL)
                except Exception as exc:
                    evidence_articles = []
                    evidence_error = f"Evidence retrieval failed: {exc}"

    evidence_articles = [_clean_article(a) for a in evidence_articles]

    # Personalised retrieval: prepend the signed-in user's knowledge-base hits.
    if rag_user_id:
        try:
            loop = asyncio.get_running_loop()
            kb_hits = await loop.run_in_executor(
                None, _rag_search, rag_user_id, "kb", req.question, 3
            )
            kb_articles = [
                _clean_article({
                    "title": h.get("title") or "Knowledge base",
                    "summary": h.get("text"),
                    "text": h.get("text"),
                    "source": h.get("source") or "Knowledge base",
                    "url": h.get("url") or None,
                    "relevance_score": h.get("score"),
                })
                for h in kb_hits
            ]
            evidence_articles = kb_articles + evidence_articles
        except Exception:
            pass

    record["news_articles"] = evidence_articles

    if req.chat_mode:
        # Conversational mode: drop the JSON-only forecast template entirely so the
        # model replies in natural language. Pass an empty template so the user
        # prompt is just the question + evidence/market context, no JSON suffix.
        system_prompt = _CHAT_SYSTEM_PROMPT
        user_prompt = build_user_prompt(record, "[question]", "full")
    else:
        user_prompt = build_user_prompt(record, prompt_text, "full")
        user_prompt += _typing_instruction(req.question_type, req.options, has_history=bool(req.history))

    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-12:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_prompt})
    return messages, evidence_articles, evidence_error


def _select_predict_provider(req: "PredictRequest"):
    # Use a user-supplied model if provided: a custom OpenAI-compatible endpoint
    # when provider_base_url is set, otherwise OpenRouter. Falls back to the
    # server default model when no key/model is given.
    alt_provider = _scads_alt_provider(req.model) if req.model else None
    if alt_provider is not None:
        # Server-hosted alternate model (allowlisted SCADS), server's own key.
        return alt_provider, _state.get("temperature", 0.0), _state.get("max_tokens", 1024)
    if req.openrouter_api_key and req.openrouter_model:
        if req.provider_base_url:
            from analyzing_llm_rationale.providers import OpenAICompatibleProvider
            provider = OpenAICompatibleProvider(
                model_name=req.openrouter_model,
                api_key=req.openrouter_api_key,
                base_url=req.provider_base_url,
            )
        else:
            from analyzing_llm_rationale.providers import OpenRouterProvider
            provider = OpenRouterProvider(
                model_name=req.openrouter_model,
                api_key=req.openrouter_api_key,
            )
        return provider, 0.7, _state.get("max_tokens", 1024)
    return _state["provider"], _state["temperature"], _state["max_tokens"]


# ── Attachment extraction ─────────────────────────────────────────────────────

def _extract_pdf_bytes(content: bytes) -> str:
    try:
        import io  # noqa: PLC0415

        from pypdf import PdfReader  # noqa: PLC0415

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF extraction failed: {exc}") from exc


def _extract_url_sync(url: str) -> Dict[str, Any]:
    import re
    from urllib.parse import urlparse
    try:
        import requests as _req
        from bs4 import BeautifulSoup
        resp = _req.get(url, timeout=20, headers={"User-Agent": "Foresea/1.0"}, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        title = (soup.title.string or "").strip() if soup.title else urlparse(url).netloc
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return {"title": title, "text": text[:20000], "source": urlparse(url).netloc, "url": url}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"URL extraction failed: {exc}") from exc


def _analytics_conn():
    _ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(_ANALYTICS_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS page_visits (
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            path TEXT,
            referrer TEXT,
            user_agent TEXT,
            timezone TEXT,
            visitor_hash TEXT
        )
        """
    )
    return conn


def _visitor_ip_ua(request: Request) -> tuple[str, str]:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip and request.client:
        ip = request.client.host
    return ip, request.headers.get("user-agent", "")


def _visitor_hash(request: Request) -> str:
    """Per-day salted visitor hash (used by the DuckDB fallback's daily uniques)."""
    ip, user_agent = _visitor_ip_ua(request)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    salt = os.environ.get("ANALYTICS_SALT", "foresea-analytics")
    raw = f"{day}:{ip}:{user_agent}:{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _visitor_id(request: Request) -> str:
    """Stable, day-independent visitor id (no raw IP stored) so cumulative
    unique-visitor counts mean *distinct people over all time*, not per-day."""
    ip, user_agent = _visitor_ip_ua(request)
    salt = os.environ.get("ANALYTICS_SALT", "foresea-analytics")
    raw = f"{ip}:{user_agent}:{salt}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def _record_visit_duckdb(event: VisitRequest, request: Request) -> None:
    conn = _analytics_conn()
    try:
        conn.execute(
            """
            INSERT INTO page_visits (path, referrer, user_agent, timezone, visitor_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                event.path,
                event.referrer,
                request.headers.get("user-agent", "")[:1000],
                event.timezone,
                _visitor_hash(request),
            ],
        )
    finally:
        conn.close()


def _record_visit_datastore(event: VisitRequest, request: Request) -> None:
    """Persist a visit in Cloud Datastore so counts survive instance recycles.

    Cumulative totals live in an ``AnalyticsStats`` singleton updated in a
    transaction; ``Visitor`` entities (keyed by the stable visitor id) dedupe
    unique people; ``PageVisit`` rows back the 30-day daily breakdown. No raw
    IP is ever stored.
    """
    client = _get_datastore()
    from google.cloud import datastore as _ds

    vid = _visitor_id(request)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    now = datetime.now(timezone.utc)

    with client.transaction():
        visit = _ds.Entity(
            client.key("PageVisit"),
            exclude_from_indexes=("referrer", "user_agent", "timezone", "path"),
        )
        visit.update(
            ts=now, day=day, path=event.path, referrer=event.referrer,
            timezone=event.timezone, visitor_id=vid,
            user_agent=request.headers.get("user-agent", "")[:1000],
        )
        client.put(visit)

        stats_key = client.key("AnalyticsStats", "global")
        stats = client.get(stats_key) or _ds.Entity(stats_key)
        visitor_key = client.key("Visitor", vid)
        is_new_visitor = client.get(visitor_key) is None
        stats["total_visits"] = int(stats.get("total_visits", 0)) + 1
        if is_new_visitor:
            stats["unique_visitors"] = int(stats.get("unique_visitors", 0)) + 1
            client.put(_ds.Entity(visitor_key))
        client.put(stats)


def _record_visit(event: VisitRequest, request: Request) -> None:
    """Record a visit in Datastore when available, else the local DuckDB."""
    if _get_datastore() is not None:
        try:
            _record_visit_datastore(event, request)
            return
        except Exception:
            logger.warning("datastore visit record failed; falling back to duckdb", exc_info=True)
    _record_visit_duckdb(event, request)


def _analytics_summary_datastore() -> "AnalyticsSummary":
    client = _get_datastore()
    stats = client.get(client.key("AnalyticsStats", "global"))
    total = int(stats["total_visits"]) if stats else 0
    unique = int(stats["unique_visitors"]) if stats else 0

    # Daily breakdown (last 30 days) from PageVisit rows; distinct counted in
    # Python over a bounded window.
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    query = client.query(kind="PageVisit")
    query.add_filter("ts", ">=", cutoff)
    by_day: Dict[str, Dict[str, Any]] = {}
    for e in query.fetch():
        d = e.get("day") or e["ts"].strftime("%Y-%m-%d")
        agg = by_day.setdefault(d, {"visits": 0, "visitors": set()})
        agg["visits"] += 1
        if e.get("visitor_id"):
            agg["visitors"].add(e["visitor_id"])
    rows = sorted(by_day.items(), reverse=True)[:30]
    return AnalyticsSummary(
        total_visits=total,
        unique_visitors=unique,
        by_day=[
            {"day": d, "visits": agg["visits"], "unique_visitors": len(agg["visitors"])}
            for d, agg in rows
        ],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Service status.",
    responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}},
)
async def health() -> Dict[str, str]:
    """Returns `{"status": "ok"}` when the server is running.

    Use this endpoint for liveness probes and uptime monitoring.
    It does **not** verify that the LLM provider is reachable.
    """
    return {"status": "ok"}


@app.get(
    "/ready",
    tags=["System"],
    summary="Readiness check",
    response_description="Whether the service is ready to serve traffic.",
)
async def ready() -> JSONResponse:
    """Returns 200 when the server is initialised and accepting traffic, 503
    otherwise (still booting, misconfigured, or draining on shutdown).

    Unlike `/health` (liveness), this verifies the forecasting provider is
    configured — use it for load-balancer readiness routing.
    """
    provider_ready = bool(_state.get("provider"))
    ok = _ready and provider_ready
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "ready": ok,
            "provider_configured": provider_ready,
            "draining": not _ready,
        },
    )


@app.post("/analytics/visit", tags=["System"], summary="Record anonymous page visit")
async def record_visit(event: VisitRequest, request: Request) -> Dict[str, str]:
    """Record one anonymous page visit.

    Stores no raw IP address. Unique visitors are estimated with a salted hash
    of IP address and user agent.
    """
    # Datastore/DuckDB I/O is blocking — keep it off the event loop.
    await asyncio.get_running_loop().run_in_executor(None, _record_visit, event, request)
    return {"status": "ok"}


@app.get("/analytics/summary", tags=["System"], response_model=AnalyticsSummary)
async def analytics_summary(request: Request) -> AnalyticsSummary:
    """Return basic page-visit counts.

    If `API_KEY` is configured, this endpoint requires `X-API-Key`.

    Counts are persisted in Cloud Datastore when available (survive deploys /
    instance recycles); otherwise they come from the local ephemeral DuckDB.
    """
    _check_api_key(request)
    if _get_datastore() is not None:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _analytics_summary_datastore
            )
        except Exception:
            logger.warning("datastore summary failed; falling back to duckdb", exc_info=True)
    conn = _analytics_conn()
    try:
        total, unique_visitors = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT visitor_hash) FROM page_visits"
        ).fetchone()
        rows = conn.execute(
            """
            SELECT
                CAST(ts AS DATE) AS day,
                COUNT(*) AS visits,
                COUNT(DISTINCT visitor_hash) AS unique_visitors
            FROM page_visits
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 30
            """
        ).fetchall()
    finally:
        conn.close()

    return AnalyticsSummary(
        total_visits=int(total or 0),
        unique_visitors=int(unique_visitors or 0),
        by_day=[
            {
                "day": str(day),
                "visits": int(visits),
                "unique_visitors": int(unique_count),
            }
            for day, visits, unique_count in rows
        ],
    )


_MARKET_CACHE_TTL = int(os.environ.get("MARKET_CACHE_TTL", "30"))


async def _fetch_market_quote(venue: str, **kwargs: Any) -> "MarketQuote":
    """Shared cache + error handling for the market-fetch endpoints."""
    from analyzing_llm_rationale import market_data

    cache_key = _cache_key("market", venue, kwargs)
    cached = _cache_get(cache_key)
    if cached is not None:
        return MarketQuote(**cached)

    loop = asyncio.get_running_loop()
    try:
        if venue == "polymarket":
            quote = await loop.run_in_executor(
                None,
                lambda: market_data.fetch_polymarket(
                    slug=kwargs.get("slug"), market_id=kwargs.get("market_id")
                ),
            )
        else:
            quote = await loop.run_in_executor(
                None, lambda: market_data.fetch_kalshi(kwargs["ticker"])
            )
    except market_data.MarketDataError as exc:
        code = 404 if "not found" in str(exc).lower() else 502
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Market provider error: {exc}") from exc

    _cache_set(cache_key, quote, _MARKET_CACHE_TTL)
    return MarketQuote(**quote)


@app.get("/markets/polymarket", tags=["Markets"], summary="Fetch a Polymarket market", response_model=MarketQuote)
async def market_polymarket(
    request: Request,
    slug: Optional[str] = None,
    id: Optional[str] = None,
) -> "MarketQuote":
    """Fetch a live Polymarket quote by market `slug` (or numeric `id`).

    The result is normalised so `probability` can be passed to `/predict` as
    `market_probability` (with `market_platform="Polymarket"`) to compute an edge.
    """
    _check_rate_limit(request)
    if not slug and not id:
        raise HTTPException(status_code=422, detail="Provide a Polymarket 'slug' or 'id'.")
    return await _fetch_market_quote("polymarket", slug=slug, market_id=id)


@app.get("/markets/kalshi", tags=["Markets"], summary="Fetch a Kalshi market", response_model=MarketQuote)
async def market_kalshi(request: Request, ticker: str) -> "MarketQuote":
    """Fetch a live Kalshi quote by market `ticker` (e.g. `KXFED-26SEP-C`)."""
    _check_rate_limit(request)
    if not ticker.strip():
        raise HTTPException(status_code=422, detail="Provide a Kalshi 'ticker'.")
    return await _fetch_market_quote("kalshi", ticker=ticker)


def _trading_http_exception(exc: Exception) -> HTTPException:
    from analyzing_llm_rationale import trading

    if isinstance(exc, trading.TradingValidationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (trading.TradingDisabledError, trading.TradingNotConfiguredError)):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, trading.TradingExecutionError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Trading error: {exc}")


@app.get(
    "/trading/accounts",
    tags=["Trading"],
    summary="Check configured trading venues",
    response_model=TradingAccountStatus,
)
async def trading_accounts(request: Request) -> TradingAccountStatus:
    """Return configured/not-configured status for server-side exchange credentials.

    This endpoint never returns private keys, API secrets, or passphrases. Live
    order submission still requires `/trading/orders` confirmation.
    """
    _check_rate_limit(request)
    _require_session(request)
    from analyzing_llm_rationale import trading

    try:
        return TradingAccountStatus(**trading.account_status())
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/preview",
    tags=["Trading"],
    summary="Preview a guarded prediction-market order",
    response_model=TradeOrderPreviewResponse,
)
async def trading_preview(req: TradeOrderRequest, request: Request) -> TradeOrderPreviewResponse:
    """Validate and normalize a Kalshi/Polymarket order without placing it."""
    _check_rate_limit(request)
    _require_session(request)
    from analyzing_llm_rationale import trading

    try:
        payload = req.model_dump(exclude_none=True)
        return TradeOrderPreviewResponse(**trading.preview_order(payload))
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.post(
    "/trading/orders",
    tags=["Trading"],
    summary="Submit a confirmed prediction-market order",
    response_model=TradeOrderResponse,
)
async def trading_order(req: TradeOrderRequest, request: Request) -> TradeOrderResponse:
    """Submit a live order after preview guardrails and exact confirmation.

    Live execution is disabled unless `FORESEA_ENABLE_TRADING=true`. Market/IOC
    style orders are separately disabled unless `FORESEA_ALLOW_MARKET_ORDERS=true`.
    """
    _check_rate_limit(request)
    claims = _require_session(request)
    from analyzing_llm_rationale import trading

    try:
        payload = req.model_dump(exclude_none=True)
        return TradeOrderResponse(**trading.place_order(payload, user_id=claims["sub"]))
    except Exception as exc:
        raise _trading_http_exception(exc) from exc


@app.post("/extract", tags=["System"], summary="Extract text from a PDF file or URL")
async def extract_attachment(  # noqa: B008
    request: Request,
    file: Optional[UploadFile] = File(None),  # noqa: B008
    url: Optional[str] = Form(None),  # noqa: B008
) -> Dict[str, Any]:
    """Extract plain text from an uploaded PDF or a web URL.

    Returns a dict compatible with the `news_articles` field of `/predict`:
    `{title, text, source, url}`. Pass one of `file` or `url`, not both.
    """
    _check_rate_limit(request)
    loop = asyncio.get_running_loop()
    if file is not None:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are supported.")
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="PDF exceeds 20 MB limit.")
        text = await loop.run_in_executor(None, _extract_pdf_bytes, content)
        return {"title": file.filename, "text": text[:20000], "source": file.filename, "url": None}
    if url:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="URL must start with http:// or https://")
        url_cache_key = _cache_key("extract_url", url)
        cached = _cache_get(url_cache_key)
        if cached is not None:
            return cached
        extracted = await loop.run_in_executor(None, _extract_url_sync, url)
        _cache_set(url_cache_key, extracted, _EXTRACT_CACHE_TTL)
        return extracted
    raise HTTPException(status_code=400, detail="Provide either a PDF file or a url.")


@app.get("/auth/config", tags=["Auth"], include_in_schema=False)
async def auth_config() -> Dict[str, str]:
    """Return public OAuth client IDs so the browser can offer sign-in options."""
    return {
        "google_client_id": _GOOGLE_CLIENT_ID or "",
        "github_client_id": _GITHUB_CLIENT_ID or "",
    }


@app.post("/auth/google", tags=["Auth"], summary="Sign in with Google", response_model=SessionResponse)
async def auth_google(req: GoogleAuthRequest) -> SessionResponse:
    """Verify a Google One-Tap ID token, create or update the user account, and return a session token.

    The browser should send the `credential` string returned by
    `google.accounts.id.initialize({ callback })` after the user grants consent.
    Store the returned `token` in `localStorage` and include it as
    `Authorization: Bearer <token>` on subsequent authenticated requests.
    """
    claims = _verify_google_token(req.credential)
    sub = str(claims["sub"])
    email = claims.get("email", "")
    name = claims.get("name", "")
    picture = claims.get("picture", "")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _upsert_user, sub, email, name, picture)
    token = _issue_session(sub, email, name, picture)
    return SessionResponse(token=token, user_id=sub, email=email, name=name, picture=picture)


@app.post("/auth/github", tags=["Auth"], summary="Sign in with GitHub", response_model=SessionResponse)
async def auth_github(req: GitHubAuthRequest) -> SessionResponse:
    """Exchange a GitHub OAuth `code` for the user's profile and return a session token.

    The browser sends users to GitHub's authorize URL, then posts the returned
    `code` (and the same `redirect_uri`) here. Store the returned `token` like the
    Google flow.
    """
    loop = asyncio.get_running_loop()
    profile = await loop.run_in_executor(None, _exchange_github_code, req.code, req.redirect_uri)
    sub, email, name, picture = profile["sub"], profile["email"], profile["name"], profile["picture"]
    await loop.run_in_executor(None, _upsert_user, sub, email, name, picture)
    token = _issue_session(sub, email, name, picture)
    return SessionResponse(token=token, user_id=sub, email=email, name=name, picture=picture)


@app.post("/auth/register", tags=["Auth"], summary="Create an account", response_model=SessionResponse)
async def auth_register(req: RegisterRequest) -> SessionResponse:
    """Create an email/password account and return a session token.

    The email becomes the account ID. Passwords are stored as salted
    PBKDF2-HMAC-SHA256 hashes; the plaintext is never persisted.
    """
    user_id = req.email
    loop = asyncio.get_running_loop()
    existing = await loop.run_in_executor(None, _get_user_record, user_id)
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    name = req.name or req.email.split("@")[0]
    password_hash = _hash_password(req.password)
    await loop.run_in_executor(None, _create_password_user, user_id, req.email, name, password_hash)
    token = _issue_session(user_id, req.email, name, "")
    return SessionResponse(token=token, user_id=user_id, email=req.email, name=name, picture="")


@app.post("/auth/login", tags=["Auth"], summary="Sign in with email and password", response_model=SessionResponse)
async def auth_login(req: LoginRequest) -> SessionResponse:
    """Verify an email/password account and return a session token."""
    user_id = req.email
    loop = asyncio.get_running_loop()
    record = await loop.run_in_executor(None, _get_user_record, user_id)
    stored_hash = record.get("password_hash") if record else None
    # Always run a verification so timing does not reveal whether the email exists.
    if not _verify_password(req.password, stored_hash or _DUMMY_PASSWORD_HASH) or not stored_hash:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    name = record.get("name") or req.email.split("@")[0]
    picture = record.get("picture", "")
    await loop.run_in_executor(None, _touch_last_login, user_id)
    token = _issue_session(user_id, req.email, name, picture)
    return SessionResponse(token=token, user_id=user_id, email=req.email, name=name, picture=picture)


@app.get("/auth/me", tags=["Auth"], summary="Get current user", response_model=AuthMeResponse)
async def auth_me(request: Request) -> AuthMeResponse:
    """Return the authenticated user decoded from the `Authorization: Bearer` session token."""
    claims = _require_session(request)
    return AuthMeResponse(
        user_id=claims["sub"],
        email=claims["email"],
        name=claims["name"],
        picture=claims["picture"],
    )


@app.get("/chat/conversations", tags=["Auth"], response_model=ChatConversationList, include_in_schema=False)
async def chat_conversations(request: Request) -> ChatConversationList:
    """List conversations for the signed-in user."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    conversations = await loop.run_in_executor(None, _list_conversations, claims["sub"])
    return ChatConversationList(conversations=[ChatConversation(**c) for c in conversations])


@app.put("/chat/conversations/{conversation_id}", tags=["Auth"], response_model=ChatConversation, include_in_schema=False)
async def save_chat_conversation(
    conversation_id: str,
    conversation: ChatConversation,
    request: Request,
) -> ChatConversation:
    """Create or replace one conversation for the signed-in user."""
    claims = _require_session(request)
    if conversation.id != conversation_id:
        raise HTTPException(status_code=400, detail="Conversation ID path/body mismatch.")
    loop = asyncio.get_running_loop()
    saved = await loop.run_in_executor(None, _put_conversation, claims["sub"], conversation.model_dump())
    return ChatConversation(**saved)


@app.delete("/chat/conversations/{conversation_id}", tags=["Auth"], include_in_schema=False)
async def delete_chat_conversation(conversation_id: str, request: Request) -> Dict[str, bool]:
    """Delete one conversation for the signed-in user."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _delete_conversation, claims["sub"], conversation_id)
    return {"ok": True}


@app.post("/rag/ingest", tags=["Knowledge"], summary="Add a document to your knowledge base")
async def rag_ingest(req: RagIngestRequest, request: Request) -> Dict[str, Any]:
    """Chunk, embed, and store a document (text or URL) in the signed-in user's
    knowledge base, so the agent can retrieve it as evidence."""
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    title, text, url = req.title, (req.text or "").strip(), (req.url or "").strip()
    source = "Knowledge base"
    if not text and url:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=422, detail="url must start with http:// or https://")
        extracted = await loop.run_in_executor(None, _extract_url_sync, url)
        text = (extracted or {}).get("text") or ""
        title = title or (extracted or {}).get("title") or url
        source = (extracted or {}).get("source") or url
    if not text:
        raise HTTPException(status_code=422, detail="Provide text or a fetchable url.")
    item = {"text": text[:200000], "title": title, "url": url, "source": source}
    try:
        added = await loop.run_in_executor(None, _rag_add, claims["sub"], req.namespace, [item])
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"chunks_added": added, "namespace": req.namespace}


@app.get("/rag/search", tags=["Knowledge"], summary="Search your knowledge base", response_model=List[RagSearchResult])
async def rag_search(request: Request, q: str, namespace: str = "kb", top_k: int = 5) -> List[RagSearchResult]:
    claims = _require_session(request)
    if not q.strip():
        raise HTTPException(status_code=422, detail="Provide a query 'q'.")
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(None, _rag_search, claims["sub"], namespace, q, max(1, min(top_k, 20)))
    return [RagSearchResult(**h) for h in hits]


@app.get("/rag/documents", tags=["Knowledge"], summary="List your knowledge-base documents")
async def rag_documents(request: Request, namespace: str = "kb") -> Dict[str, Any]:
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, _rag_documents, claims["sub"], namespace)
    return {"namespace": namespace, "documents": docs}


@app.delete("/rag/documents", tags=["Knowledge"], summary="Delete knowledge-base documents")
async def rag_delete(request: Request, namespace: str = "kb", doc_id: Optional[str] = None) -> Dict[str, int]:
    claims = _require_session(request)
    loop = asyncio.get_running_loop()
    removed = await loop.run_in_executor(None, _rag_delete, claims["sub"], namespace, doc_id)
    return {"removed": removed}


@app.post(
    "/predict",
    tags=["Inference"],
    summary="Run a single forecasting prediction",
    response_description="Prediction result with confidence score, rationale, and evidence sources.",
    responses={
        200: {"description": "Prediction returned successfully."},
        400: {"description": "Invalid request — unknown variant or malformed input."},
        401: {"description": "Missing or invalid `X-API-Key` header (only when API key is configured)."},
        429: {"description": "Rate limit exceeded. Retry after 60 seconds."},
        503: {"description": "Server not yet initialised — LLM provider not loaded."},
    },
    response_model=PredictResponse,
)
async def predict(req: PredictRequest, request: Request = None, kb_user_id: Optional[str] = None) -> PredictResponse:
    """Submit a forecasting question and receive a typed structured prediction.

    The model returns:
    - **`question_type`** — `binary`, `multiple_choice`, `numeric`, or `date`
    - **`predicted_answer`** — `Yes`/`No`, top option, or median estimate
    - **`confidence`** — probability (0–1) for binary and multiple-choice answers
    - **`options`** — per-option probabilities for multiple-choice questions
    - **`range_forecast`** — p10/p50/p90 bounds for numeric and date questions
    - **`rationale`** — 2–4 sentence explanation
    - **`evidence_sources`** — news articles used as context
    - **`market_analysis`** — model-vs-market edge when `market_probability` is provided

    ### Choosing a variant

    Start with `variant0_neutral_baseline` (the default). Switch to other variants
    to inject additional structure into the prompt — for example, `variant3_reasoning_type`
    asks the model to identify whether the prediction is based on speculation, an expert
    forecast, or a stated plan.

    ### Providing your own evidence

    Pass pre-fetched articles in `news_articles` to use them directly.
    Leave the list empty to trigger automatic news retrieval (if the evidence
    pipeline is configured on the server).

    ### Example — binary request

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "Will oil prices exceed $100 per barrel in 2026?",
        "question_type": "binary",
        "market_platform": "Kalshi",
        "market_probability": 0.31
      }'
    ```

    ### Example — multiple choice

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "Who will win the 2026 Formula 1 drivers championship?",
        "question_type": "multiple_choice",
        "options": ["Max Verstappen", "Lando Norris", "Charles Leclerc", "Lewis Hamilton", "Other"],
        "attach_evidence": false
      }'
    ```

    ### Example — numeric forecast with context

    ```bash
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "question": "What will US CPI inflation be in December 2026?",
        "question_type": "numeric",
        "resolution_criteria": "Use the year-over-year CPI-U inflation rate for December 2026.",
        "categories": ["Economics", "United States"],
        "variant": "variant5_key_conditions"
      }'
    ```
    """
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    variants = _state["variants"]
    if req.variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{req.variant}'. Valid: {sorted(variants)}",
        )

    # Signed-in users get personalised (knowledge-base) evidence, so their
    # responses must never be shared via the cache.
    rag_user_id = kb_user_id or _optional_user_id(request)

    # Serve identical, non-personalised forecasts from cache to cut latency and
    # model spend. History, BYOK, or a signed-in user disable caching.
    cacheable = (
        _PREDICT_CACHE_TTL > 0
        and not req.history
        and not req.openrouter_api_key
        and not rag_user_id
    )
    predict_cache_key = None
    prompt_date = datetime.now(timezone.utc).date().isoformat()
    if cacheable:
        predict_cache_key = _cache_key(
            "predict",
            {
                "prompt_date": prompt_date,
                "question": req.question,
                "description": req.description,
                "variant": req.variant,
                "question_type": req.question_type,
                "options": req.options,
                "chat_mode": req.chat_mode,
                "attach_evidence": req.attach_evidence,
                "evidence_top_k": req.evidence_top_k,
                "news_articles": [a.model_dump() for a in req.news_articles],
                "market_platform": req.market_platform,
                "market_url": req.market_url,
                "market_outcome": req.market_outcome,
                "market_probability": req.market_probability,
                "model_key": _state.get("model_key"),
                "temperature": _state.get("temperature"),
            },
        )
        cached = _cache_get(predict_cache_key)
        if cached is not None:
            return PredictResponse(**cached)

    messages, evidence_articles, evidence_error = await _prepare_predict_messages(req, rag_user_id)
    provider, temperature, max_tokens = _select_predict_provider(req)

    try:
        content = await _provider_chat(provider, messages, temperature, max_tokens)
    except Exception as exc:
        raise _provider_http_error(exc) from exc

    parsed = _parse_json_dict(content)
    if req.chat_mode:
        text = content.strip()
        model_key = _model_key_for_request(req)
        response = PredictResponse(
            question_type="chat",
            rationale=text, model_rationale=text,
            variant=req.variant, model_key=model_key,
            evidence_sources=_evidence_sources(evidence_articles),
            evidence_articles=_news_articles(evidence_articles),
            evidence_error=evidence_error,
        )
    else:
        response = _build_typed_response(req, parsed, content, evidence_articles, evidence_error)

    if predict_cache_key is not None:
        _cache_set(predict_cache_key, response.model_dump(), _PREDICT_CACHE_TTL)

    # Best-effort: index the forecast for "search my past forecasts", but only if
    # the embedder is already loaded — never pay a cold start on the forecast path.
    if rag_user_id and rag.is_loaded():
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _rag_add, rag_user_id, "forecasts", [{
                "text": f"Q: {req.question}\nForecast: {response.predicted_answer or response.question_type} — "
                        f"{response.model_rationale or response.rationale or ''}",
                "title": req.question[:300], "source": "Past forecast",
            }])
        except Exception:
            pass
    return response


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post(
    "/predict/stream",
    tags=["Inference"],
    summary="Stream a conversational forecasting response",
    response_description="Server-sent events containing model text deltas and a final PredictResponse payload.",
)
async def predict_stream(req: PredictRequest, request: Request) -> StreamingResponse:
    """Stream chat-mode model output as server-sent events.

    The final `done` event contains the same `PredictResponse` shape that `/predict`
    returns for chat-mode requests. Structured forecast calls should keep using
    `/predict`; this endpoint forces conversational output.
    """
    _check_rate_limit(request)
    _check_api_key(request)

    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    stream_req = req.model_copy(update={"chat_mode": True})
    variants = _state["variants"]
    if stream_req.variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variant '{stream_req.variant}'. Valid: {sorted(variants)}",
        )

    rag_user_id = _optional_user_id(request)

    async def events():
        yield _sse_event("meta", {
            "status": "preparing",
            "question_type": "chat",
            "variant": stream_req.variant,
            "model_key": _model_key_for_request(stream_req),
        })
        try:
            messages, evidence_articles, evidence_error = await _prepare_predict_messages(
                stream_req, rag_user_id
            )
            provider, temperature, max_tokens = _select_predict_provider(stream_req)
        except HTTPException as exc:
            yield _sse_event("error", {"status_code": exc.status_code, "detail": exc.detail})
            return
        except Exception:
            logger.exception("predict stream setup failed")
            yield _sse_event("error", {
                "status_code": 500,
                "detail": "The streaming request could not be prepared.",
            })
            return

        yield _sse_event("meta", {
            "status": "streaming",
            "question_type": "chat",
            "variant": stream_req.variant,
            "model_key": _model_key_for_request(stream_req),
            "evidence_sources": [s.model_dump(mode="json") for s in _evidence_sources(evidence_articles)],
            "evidence_articles": [a.model_dump(mode="json") for a in _news_articles(evidence_articles)],
            "evidence_error": evidence_error,
        })

        chunks: List[str] = []
        try:
            async for chunk in _provider_stream_chat(provider, messages, temperature, max_tokens):
                if await request.is_disconnected():
                    return
                chunks.append(chunk)
                yield _sse_event("delta", {"text": chunk})
        except Exception as exc:
            http_exc = _provider_http_error(exc)
            yield _sse_event("error", {
                "status_code": http_exc.status_code,
                "detail": http_exc.detail,
            })
            return

        text = "".join(chunks).strip()
        response = PredictResponse(
            question_type="chat",
            rationale=text,
            model_rationale=text,
            variant=stream_req.variant,
            model_key=_model_key_for_request(stream_req),
            evidence_sources=_evidence_sources(evidence_articles),
            evidence_articles=_news_articles(evidence_articles),
            evidence_error=evidence_error,
        )
        yield _sse_event("done", {"response": response.model_dump(mode="json")})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/vertex-predict",
    tags=["Inference"],
    summary="Vertex AI batch prediction (instances wrapper)",
    response_description="Array of prediction results, one per input instance.",
    responses={
        200: {"description": "All predictions returned successfully."},
        400: {"description": "One or more instances are malformed."},
        429: {"description": "Rate limit exceeded."},
        503: {"description": "Server not yet initialised."},
    },
    response_model=VertexPredictResponse,
    include_in_schema=False,
)
async def vertex_predict(req: VertexPredictRequest, request: Request = None) -> VertexPredictResponse:
    """Vertex AI-compatible prediction endpoint.

    Wraps `/predict` in the Vertex AI contract:
    - **Request**: `{"instances": [PredictRequest, ...]}`
    - **Response**: `{"predictions": [PredictResponse, ...]}`

    Instances are processed sequentially. Maximum 10 instances per call.

    This endpoint is called automatically by the Vertex AI SDK and REST API.
    For direct use, prefer `/predict` instead.

    ```python
    from google.cloud import aiplatform

    endpoint = aiplatform.Endpoint("projects/.../endpoints/7325853011580813312")
    response = endpoint.predict(instances=[
        {"question": "Will oil prices exceed $100 per barrel in 2025?"}
    ])
    ```
    """
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)
    predictions = []
    for instance in req.instances:
        result = await predict(PredictRequest(**instance))
        predictions.append(result.model_dump())
    return VertexPredictResponse(predictions=predictions)


# ── Agent: orchestrated, autonomous analysis ─────────────────────────────────

_AGENT_SKILL_SYSTEM = (
    "You are one analysis skill inside a forecasting agent. Apply the requested "
    "skill to the question, the agent's current forecast, and the evidence. "
    "Respond in 2-5 sentences of clear natural language (light markdown is fine). "
    "Do not output JSON or restate the whole forecast — add the specific insight "
    "the skill asks for."
)


# Alternate server-hosted SCADS models the public API may forecast with (using
# the server's own key) — for the multi-model paper-trading comparison.
_SCADS_BASE_URL = os.environ.get("SCADS_BASE_URL", "https://llm.scads.ai/v1/chat/completions")
_SCADS_MODEL_ALLOWLIST = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "kimi-k2.6": "moonshotai/Kimi-K2.6",
}


def _scads_alt_provider(model_label: str):
    """Build a provider for an allowlisted alternate SCADS model using the server's
    own key. Returns None if the label is the server default or not allowlisted."""
    label = (model_label or "").strip()
    if not label or label == _state.get("model_key") or label not in _SCADS_MODEL_ALLOWLIST:
        return None
    scads_key = os.environ.get("SCADS_AI_API_KEY")
    if not scads_key:
        raise HTTPException(status_code=503, detail="Alternate models are not configured on this server.")
    from analyzing_llm_rationale.providers import OpenAICompatibleProvider
    return OpenAICompatibleProvider(
        model_name=_SCADS_MODEL_ALLOWLIST[label], api_key=scads_key, base_url=_SCADS_BASE_URL)


def _select_provider(
    openrouter_api_key: Optional[str],
    openrouter_model: Optional[str],
    provider_base_url: Optional[str],
):
    """Return (provider, temperature, max_tokens): BYOK model if given, else the server default."""
    if openrouter_api_key and openrouter_model:
        max_tokens = _state.get("max_tokens", 1024)
        if provider_base_url:
            from analyzing_llm_rationale.providers import OpenAICompatibleProvider
            return (
                OpenAICompatibleProvider(
                    model_name=openrouter_model, api_key=openrouter_api_key, base_url=provider_base_url
                ),
                0.7,
                max_tokens,
            )
        from analyzing_llm_rationale.providers import OpenRouterProvider
        return OpenRouterProvider(model_name=openrouter_model, api_key=openrouter_api_key), 0.7, max_tokens
    return _state["provider"], _state["temperature"], _state["max_tokens"]


def _parse_market_url(text: str) -> Optional[tuple]:
    """Extract (venue, kind, identifier) from a Polymarket/Kalshi URL in `text`.

    Polymarket: the slug is the last path segment
    (e.g. .../lpl/lol-al-we-2026-06-01 -> slug). Kalshi: the segment after
    /markets/ is the ticker. Returns None if no recognised URL is present.
    """
    for token in re.findall(r"https?://\S+", text or ""):
        parsed = urlparse(token)
        host = (parsed.hostname or "").lower()
        parts = [p for p in parsed.path.split("/") if p]
        if not parts:
            continue
        if "polymarket.com" in host:
            return ("polymarket", "slug", parts[-1])
        if "kalshi.com" in host and "markets" in parts:
            i = parts.index("markets")
            if i + 1 < len(parts):
                return ("kalshi", "ticker", parts[i + 1])
    return None


def _agent_recommendation(edge: Optional[float], outcome: str) -> tuple[str, str]:
    """Deterministic call from the model-vs-market edge."""
    out = outcome or "Yes"
    if edge is None:
        return "no_market_price", "No market price supplied, so no edge could be computed."
    pts = round(abs(edge) * 100)
    if abs(edge) < 0.05:
        return "hold", f"Model is within {pts} pts of the market on {out} — roughly fair value."
    if edge > 0:
        return "buy_yes", f"Model is {pts} pts above the market on {out}; {out} looks underpriced."
    return "buy_no", f"Model is {pts} pts below the market on {out}; {out} looks overpriced."


async def _run_agent_skill(skill: AgentSkill, context: str, provider, temperature, max_tokens) -> AgentSkillResult:
    messages = [
        {"role": "system", "content": _AGENT_SKILL_SYSTEM},
        {"role": "user", "content": f"{context}\n\nSkill: {skill.name}\nInstruction: {skill.instruction}"},
    ]
    try:
        output = await _provider_chat(provider, messages, temperature, max_tokens)
        return AgentSkillResult(name=skill.name, output=(output or "").strip())
    except Exception as exc:
        logger.warning("agent skill %r failed: %s", skill.name, type(exc).__name__)
        return AgentSkillResult(name=skill.name, output="(this analysis step is temporarily unavailable)")


@app.post("/agent/analyze", tags=["Agents"], summary="Run the analysis agent on a live question", response_model=AgentReport)
async def agent_analyze(req: AgentAnalyzeRequest, request: Request = None) -> AgentReport:
    """Orchestrate an end-to-end analysis of a live market question.

    Pipeline: resolve the market (fetch a live Polymarket/Kalshi price when an
    identifier is given) → gather evidence and forecast → compute the model-vs-market
    edge → run any custom **skills** → recommend. Returns one structured report.
    """
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    pipeline: List[str] = []

    # 1. Resolve the market (live price). Explicit identifiers win; otherwise try
    #    to pull a Polymarket/Kalshi URL out of the question text.
    quote: Optional[MarketQuote] = None
    venue = (req.platform or "").strip().lower()
    slug, market_id, ticker = req.slug, req.market_id, req.ticker
    from_url = False
    if not (venue and (slug or market_id or ticker)):
        parsed = _parse_market_url(req.question or "")
        if parsed:
            venue, kind, ident = parsed
            slug = ident if kind == "slug" else None
            ticker = ident if kind == "ticker" else None
            from_url = True
    if venue and (slug or market_id or ticker):
        try:
            if "poly" in venue:
                quote = await _fetch_market_quote("polymarket", slug=slug, market_id=market_id)
            elif "kalshi" in venue:
                quote = await _fetch_market_quote("kalshi", ticker=ticker)
        except HTTPException:
            if not from_url:
                raise  # explicit identifier that didn't resolve -> surface the error
        if quote is not None:
            pipeline.append("resolve_market")

    # Prefer the market's real question when the user only pasted a link.
    question = (req.question or "").strip()
    if quote is not None and (not question or _parse_market_url(question)):
        question = quote.question or question
    if not question:
        raise HTTPException(status_code=422, detail="Provide a question, or a platform plus market identifier.")

    # Optional: self-calibration grounding from the live track record.
    grounding_note = None
    if req.ground_in_record:
        agg = await asyncio.get_running_loop().run_in_executor(None, _read_live_track_record)
        grounding_note = agent_capabilities.build_grounding_note(agg) or None
        if grounding_note:
            pipeline.append("ground_in_record")

    # Optional: ReAct tool-using loop instead of the fixed pipeline below.
    if req.tool_loop:
        return await _agent_tool_loop(req, request, question, quote, grounding_note)

    # Feed the calibration note to the forecast as a prior (via history, so it
    # doesn't suppress live evidence retrieval the way news_articles would).
    history = list(req.history)
    if grounding_note:
        history = history + [{"role": "user",
                              "content": f"[Self-calibration context — apply as a prior, not a hard rule]\n{grounding_note}"}]

    # 2. Evidence + forecast + edge — reuse the /predict pipeline.
    pred_req = PredictRequest(
        question=question,
        attach_evidence=True,
        evidence_top_k=req.evidence_top_k,
        variant=req.variant,
        history=history,
        market_platform=(quote.platform if quote else req.platform),
        market_url=(quote.market_url if quote else None),
        market_outcome=(quote.outcome if quote else None),
        market_probability=(quote.probability if quote else req.market_probability),
        openrouter_api_key=req.openrouter_api_key,
        openrouter_model=req.openrouter_model,
        provider_base_url=req.provider_base_url,
    )
    result = await predict(pred_req, kb_user_id=_optional_user_id(request))
    pipeline.extend(["gather_evidence", "forecast"])

    analysis = result.market_analysis
    edge = analysis.edge if analysis else None
    model_probability = analysis.model_probability if analysis else result.confidence
    stance = analysis.stance if analysis else None
    outcome = (quote.outcome if quote else None) or "Yes"
    recommendation, detail = _agent_recommendation(edge, outcome)
    if analysis is not None:
        pipeline.append("price_edge")

    # 3. Skills — the built-in forecasting toolkit (optional) plus the caller's
    #    own custom steps, all run over the forecast context.
    skills_to_run: List[AgentSkill] = []
    if req.builtin_skills:
        skills_to_run.extend(AgentSkill(**s) for s in agent_capabilities.builtin_skills())
    skills_to_run.extend(req.skills)
    skill_results: List[AgentSkillResult] = []
    if skills_to_run:
        provider, temperature, max_tokens = _select_provider(
            req.openrouter_api_key, req.openrouter_model, req.provider_base_url
        )
        sources_txt = "\n".join(
            f"- {s.source}: {s.title}" for s in result.evidence_sources[:8]
        ) or "(no evidence retrieved)"
        context = (
            f"Question: {question}\n"
            f"Forecast: {result.predicted_answer} "
            f"(confidence {result.confidence if result.confidence is not None else 'n/a'})\n"
            f"Market-implied probability: {pred_req.market_probability}\n"
            f"Thesis: {result.model_rationale or result.rationale or ''}\n"
            f"Evidence:\n{sources_txt}"
        )
        skill_results = await asyncio.gather(
            *(_run_agent_skill(s, context, provider, temperature, max_tokens) for s in skills_to_run)
        )
        pipeline.append("skills")

    pipeline.append("recommend")

    return AgentReport(
        question=question,
        pipeline=pipeline,
        platform=(quote.platform if quote else req.platform),
        market_url=(quote.market_url if quote else None),
        outcome=outcome,
        market_probability=(analysis.market_probability if analysis else None),
        model_probability=model_probability,
        edge=edge,
        stance=stance,
        recommendation=recommendation,
        recommendation_detail=detail,
        confidence=result.confidence,
        question_type=result.question_type,
        thesis=result.model_rationale or result.rationale or "",
        evidence_sources=result.evidence_sources,
        skills=list(skill_results),
        grounding=grounding_note,
    )


async def _agent_tool_loop(req: "AgentAnalyzeRequest", request, question: str,
                           quote: "Optional[MarketQuote]", grounding_note: Optional[str]) -> "AgentReport":
    """ReAct tool-using loop: the model plans and calls tools (forecast, market
    fetch, evidence search, venue scan, track record), then answers. Falls back
    cleanly to a no-edge report if no forecast tool was used."""
    from analyzing_llm_rationale import market_data

    provider, temperature, max_tokens = _select_provider(
        req.openrouter_api_key, req.openrouter_model, req.provider_base_url)
    loop = asyncio.get_running_loop()
    last: Dict[str, Any] = {}

    async def _tool_forecast(args):
        q = str(args.get("question") or question)
        mp = args.get("market_probability")
        r = await predict(PredictRequest(
            question=q, attach_evidence=True, evidence_top_k=req.evidence_top_k, variant=req.variant,
            market_probability=mp, openrouter_api_key=req.openrouter_api_key,
            openrouter_model=req.openrouter_model, provider_base_url=req.provider_base_url),
            kb_user_id=_optional_user_id(request))
        a = r.market_analysis
        last.update(answer=r.predicted_answer, confidence=r.confidence,
                    model_probability=(a.model_probability if a else r.confidence),
                    market_probability=(a.market_probability if a else mp),
                    edge=(a.edge if a else None), stance=(a.stance if a else None),
                    thesis=r.model_rationale or r.rationale or "", question_type=r.question_type,
                    evidence_sources=list(r.evidence_sources))
        msg = f"Forecast: {r.predicted_answer} (confidence {r.confidence})."
        if a and a.edge is not None:
            msg += (f" Model {round((a.model_probability or 0) * 100)}% vs market "
                    f"{round((a.market_probability or 0) * 100)}%, edge {round(a.edge * 100):+d} pts.")
        return msg + " " + (r.model_rationale or r.rationale or "")[:600]

    async def _tool_get_market(args):
        plat = str(args.get("platform", "")).lower()
        try:
            if "poly" in plat:
                q = await _fetch_market_quote("polymarket", slug=args.get("slug"), market_id=args.get("market_id"))
            elif "kalshi" in plat:
                q = await _fetch_market_quote("kalshi", ticker=args.get("ticker"))
            else:
                return "Specify platform 'polymarket' or 'kalshi' plus a slug/ticker."
        except HTTPException as exc:
            return f"(market fetch failed: {exc.detail})"
        return f"{q.platform}: {q.question} — {q.outcome} at {round((q.probability or 0) * 100)}% ({q.market_url})"

    async def _tool_search_evidence(args):
        ep = _state.get("evidence_pipeline")
        if ep is None:
            return "(evidence retrieval not configured)"
        query = str(args.get("query") or question)
        try:
            arts = await loop.run_in_executor(None, lambda: ep.fetch_summarize_rank(query, top_k=5))
        except Exception:
            return "(evidence retrieval failed)"
        return "\n".join(f"- {a.get('source', '?')}: {a.get('title', '')}" for a in (arts or [])[:6]) or "No evidence found."

    async def _tool_scan(args):
        plat = str(args.get("platform", "polymarket")).lower()
        fn = market_data.list_polymarket if "poly" in plat else market_data.list_kalshi if "kalshi" in plat else None
        if fn is None:
            return "platform must be 'polymarket' or 'kalshi'"
        try:
            qs = await loop.run_in_executor(None, lambda: fn(limit=5, query=args.get("query")))
        except Exception:
            return "(listing failed)"
        return "\n".join(f"- [{q.get('platform')}] {(q.get('question') or '')[:70]} @ "
                         f"{round((q.get('probability') or 0) * 100)}%" for q in qs) or "No markets found."

    async def _tool_track_record(args):
        agg = await loop.run_in_executor(None, _read_live_track_record)
        return agent_capabilities.build_grounding_note(agg) or "No resolved forecasts yet."

    tools = {"forecast": _tool_forecast, "get_market": _tool_get_market,
             "search_evidence": _tool_search_evidence, "scan_markets": _tool_scan,
             "track_record": _tool_track_record}
    specs = [
        {"name": "forecast", "args": "question, market_probability?", "description": "Produce a probability forecast (with evidence) for a question; pass market_probability to get the edge."},
        {"name": "get_market", "args": "platform, slug|ticker", "description": "Fetch a live Polymarket/Kalshi price."},
        {"name": "search_evidence", "args": "query", "description": "Retrieve recent news headlines relevant to a query."},
        {"name": "scan_markets", "args": "platform, query?", "description": "List live markets on a venue (optionally filtered by keyword)."},
        {"name": "track_record", "args": "", "description": "Get the model's own live calibration / skill-vs-market."},
    ]

    async def chat_fn(messages):
        return await _provider_chat(provider, messages, temperature, max_tokens)

    q = question
    if grounding_note:
        q = f"{question}\n\n[Self-calibration context]\n{grounding_note}"
    rule = ("You MUST call the `forecast` tool before your final answer — it produces "
            "the probability and edge the report needs. Never state a probability "
            "without having called `forecast` for it. For example, after finding a "
            "market with scan_markets/get_market, call forecast with that market's "
            "question and its market_probability, then base your answer on that result.")
    backstopped = False
    try:
        res = await agent_capabilities.run_tool_loop(
            q, tools, specs, chat_fn, max_steps=req.max_tool_steps, extra_rules=rule)
        # Deterministic backstop: if the model answered without ever calling
        # `forecast`, run it ourselves so edge/recommendation always populate.
        if not last:
            backstopped = True
            await _tool_forecast({"question": question,
                                  "market_probability": (quote.probability if quote else req.market_probability)})
    except Exception as exc:
        raise _provider_http_error(exc) from exc

    outcome = (quote.outcome if quote else None) or "Yes"
    edge = last.get("edge")
    recommendation, detail = _agent_recommendation(edge, outcome)
    pipeline = ["tool_loop"] + (["forecast"] if last else [])
    if grounding_note:
        pipeline.insert(0, "ground_in_record")
    return AgentReport(
        question=question, pipeline=pipeline,
        platform=(quote.platform if quote else req.platform),
        market_url=(quote.market_url if quote else None),
        outcome=outcome, market_probability=last.get("market_probability"),
        model_probability=last.get("model_probability"), edge=edge, stance=last.get("stance"),
        recommendation=recommendation, recommendation_detail=detail, confidence=last.get("confidence"),
        question_type=last.get("question_type", "binary"),
        # When we had to backstop the forecast, the model's prose can cite a
        # different number than the structured forecast that drives the
        # recommendation — so use the forecast's own rationale to stay consistent.
        thesis=(last.get("thesis") if backstopped and last.get("thesis") else res.get("answer", "")),
        evidence_sources=last.get("evidence_sources", []),
        grounding=grounding_note, tool_transcript=res.get("transcript", []),
    )


@app.get("/agent/scan", tags=["Agents"], summary="Scan a venue for mispriced markets", response_model=AgentScanResponse)
async def agent_scan(
    request: Request = None,
    platform: str = "polymarket",
    limit: int = 4,
    min_edge: float = 0.1,
    evidence_top_k: int = 3,
    query: Optional[str] = None,
) -> AgentScanResponse:
    """Scan live markets on one or both venues and surface the most mispriced.

    Lists active markets (Polymarket and/or Kalshi), forecasts each, compares
    against the market price, and returns those whose model-vs-market gap is at
    least `min_edge`. `platform` accepts `polymarket`, `kalshi`, or `all`/`both`.
    `query` optionally filters to markets whose question contains that keyword
    (e.g. `nba`). Bounded by `limit` per venue (max 8) since each market runs a
    full forecast; results are cached briefly.
    """
    if request is not None:
        _check_rate_limit(request)
        _check_api_key(request)
    if not _state:
        raise HTTPException(status_code=503, detail="Server not initialised")

    venue = (platform or "polymarket").strip().lower()
    kw = (query or "").strip() or None
    limit = max(1, min(limit, 8))
    evidence_top_k = max(1, min(evidence_top_k, 6))

    if venue in ("all", "both") or ("poly" in venue and "kalshi" in venue):
        venues = ["polymarket", "kalshi"]
    elif "kalshi" in venue:
        venues = ["kalshi"]
    elif "poly" in venue:
        venues = ["polymarket"]
    else:
        raise HTTPException(status_code=422, detail="platform must be 'polymarket', 'kalshi', or 'all'.")

    cache_key = _cache_key("scan", ",".join(venues), kw or "", limit, round(min_edge, 3),
                           evidence_top_k, _state.get("model_key"))
    cached = _cache_get(cache_key)
    if cached is not None:
        return AgentScanResponse(**cached)

    from analyzing_llm_rationale import market_data
    loop = asyncio.get_running_loop()
    _listers = {"polymarket": market_data.list_polymarket, "kalshi": market_data.list_kalshi}
    quotes: List[Dict[str, Any]] = []
    list_errors: List[str] = []
    for v in venues:
        try:
            vq = await loop.run_in_executor(None, lambda fn=_listers[v]: fn(limit=limit, query=kw))
            quotes.extend(vq[:limit])
        except market_data.MarketDataError as exc:
            list_errors.append(f"{v}: {exc}")
    if not quotes:
        if list_errors and len(list_errors) == len(venues):
            raise HTTPException(status_code=502, detail=f"Could not list markets: {'; '.join(list_errors)}")
        return AgentScanResponse(platform=" + ".join(p.title() for p in venues), scanned=0, opportunities=[])
    platform_name = " + ".join(p.title() for p in venues)

    async def _score(quote: Dict[str, Any]) -> Optional[ScanOpportunity]:
        try:
            res = await predict(PredictRequest(
                question=quote["question"],
                attach_evidence=True,
                evidence_top_k=evidence_top_k,
                market_platform=quote.get("platform"),
                market_url=quote.get("market_url"),
                market_outcome=quote.get("outcome"),
                market_probability=quote.get("probability"),
            ))
        except Exception:
            return None
        analysis = res.market_analysis
        if analysis is None or analysis.edge is None:
            return None
        recommendation, _ = _agent_recommendation(analysis.edge, quote.get("outcome") or "Yes")
        return ScanOpportunity(
            question=quote["question"],
            market_url=quote.get("market_url"),
            outcome=quote.get("outcome") or "Yes",
            market_probability=analysis.market_probability,
            model_probability=analysis.model_probability,
            edge=analysis.edge,
            stance=analysis.stance,
            recommendation=recommendation,
        )

    scored = await asyncio.gather(*(_score(q) for q in quotes))
    opportunities = [s for s in scored if s is not None and s.edge is not None and abs(s.edge) >= min_edge]
    opportunities.sort(key=lambda o: abs(o.edge), reverse=True)
    response = AgentScanResponse(platform=platform_name, scanned=len(quotes), opportunities=opportunities)
    _cache_set(cache_key, response.model_dump(), _MARKET_CACHE_TTL * 10)
    return response


# The live track record is advanced by a GitHub Action (see
# .github/workflows/track-record-tick.yml + scripts/track_record_tick.py), which
# commits the public aggregate to static/track_record_live.json. The server only
# *serves* that result (see _read_live_track_record + GET /track-record) — no
# batch work runs on Cloud Run, which is what kept OOM/timeout-failing.
