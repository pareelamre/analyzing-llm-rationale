"""Outbound PR-agent outreach to explicit agent endpoints.

This is deliberately not wired to the public unauthenticated Foresea API. It is
for operator-run outreach from an explicit target list, with dry-run as the
default and a required ``--send`` flag in the script wrapper.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from analyzing_llm_rationale.pr_agent import build_pr_agent_packet


@dataclass(frozen=True)
class OutreachTarget:
    name: str
    endpoint: str
    audience: str = "agent"
    transport: str = "webhook"
    headers: Optional[Dict[str, str]] = None
    form: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None


def load_targets(path: Path) -> List[OutreachTarget]:
    """Load outreach targets from JSON.

    Accepted shapes:
    - ``[{...}, {...}]``
    - ``{"targets": [{...}, {...}]}``
    """
    raw = json.loads(path.read_text())
    rows = raw.get("targets") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("target file must be a list or an object with a 'targets' list")
    targets = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"target {i} must be an object")
        targets.append(_target_from_dict(row, i))
    return targets


def _target_from_dict(row: Dict[str, Any], index: int) -> OutreachTarget:
    name = str(row.get("name") or "").strip()
    endpoint = str(row.get("endpoint") or "").strip()
    if not name:
        raise ValueError(f"target {index} is missing name")
    if not endpoint:
        raise ValueError(f"target {name!r} is missing endpoint")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"target {name!r} endpoint must be an http(s) URL")
    transport = str(row.get("transport") or "webhook").strip().lower()
    if transport not in {"webhook", "form", "json"}:
        raise ValueError(f"target {name!r} transport must be 'webhook', 'form', or 'json'")
    headers = _expand_headers(row.get("headers"), target_name=name)
    form = _expand_form(row.get("form"), target_name=name) if transport == "form" else None
    body = _expand_body(row.get("body"), target_name=name) if transport == "json" else None
    if transport == "form" and not form:
        raise ValueError(f"target {name!r} form transport requires a form object")
    if transport == "json" and not body:
        raise ValueError(f"target {name!r} json transport requires a body object")
    return OutreachTarget(
        name=name,
        endpoint=endpoint,
        audience=str(row.get("audience") or "agent").strip().lower(),
        transport=transport,
        headers=headers,
        form=form,
        body=body,
    )


def _expand_headers(headers: Any, *, target_name: str) -> Optional[Dict[str, str]]:
    if headers is None:
        return None
    if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ValueError(f"target {target_name!r} headers must be an object of strings")
    expanded: Dict[str, str] = {}
    for key, value in headers.items():
        if value.startswith("$"):
            env_name = value[1:]
            secret = os.environ.get(env_name)
            if not secret:
                raise ValueError(f"target {target_name!r} header {key!r} references unset env var {env_name!r}")
            expanded[key] = secret
        else:
            expanded[key] = value
    return expanded


def _expand_form(form: Any, *, target_name: str) -> Optional[Dict[str, Any]]:
    if form is None:
        return None
    if not isinstance(form, dict) or not all(isinstance(k, str) for k in form):
        raise ValueError(f"target {target_name!r} form must be an object")
    expanded: Dict[str, Any] = {}
    for key, value in form.items():
        if isinstance(value, str) and value.startswith("$"):
            env_name = value[1:]
            secret = os.environ.get(env_name)
            if not secret:
                raise ValueError(f"target {target_name!r} form field {key!r} references unset env var {env_name!r}")
            expanded[key] = secret
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            expanded[key] = value
        elif isinstance(value, str):
            expanded[key] = value
        else:
            raise ValueError(f"target {target_name!r} form field {key!r} must be a string or list of strings")
    return expanded


def _expand_body(body: Any, *, target_name: str) -> Optional[Dict[str, Any]]:
    if body is None:
        return None
    if not isinstance(body, dict) or not all(isinstance(k, str) for k in body):
        raise ValueError(f"target {target_name!r} body must be an object")
    return {key: _expand_json_value(value, target_name=target_name, path=key) for key, value in body.items()}


def _expand_json_value(value: Any, *, target_name: str, path: str) -> Any:
    if isinstance(value, str):
        if value.startswith("$"):
            env_name = value[1:]
            secret = os.environ.get(env_name)
            if not secret:
                raise ValueError(
                    f"target {target_name!r} body field {path!r} references unset env var {env_name!r}"
                )
            return secret
        return value
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError(f"target {target_name!r} body field {path!r} must use string keys")
        return {
            key: _expand_json_value(child, target_name=target_name, path=f"{path}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_json_value(child, target_name=target_name, path=f"{path}[]")
            for child in value
        ]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(
        f"target {target_name!r} body field {path!r} must be JSON-compatible"
    )


def target_key(target: OutreachTarget) -> str:
    """Stable dedupe key for send-once state."""
    return f"{target.name}|{target.endpoint}"


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"sent": {}}
    text = path.read_text().strip()
    if not text:
        return {"sent": {}}
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("outreach state must be an object")
    sent = raw.get("sent")
    if not isinstance(sent, dict):
        raw["sent"] = {}
    return raw


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def filter_unsent(targets: Iterable[OutreachTarget], state: Dict[str, Any], *, resend: bool = False) -> List[OutreachTarget]:
    if resend:
        return list(targets)
    sent = state.get("sent") if isinstance(state, dict) else {}
    sent = sent if isinstance(sent, dict) else {}
    return [target for target in targets if target_key(target) not in sent]


def mark_sent(state: Dict[str, Any], results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    sent = state.setdefault("sent", {})
    now = datetime.now(timezone.utc).isoformat()
    for result in results:
        if result.get("status") != "sent":
            continue
        key = f"{result.get('target')}|{result.get('endpoint')}"
        sent[key] = {
            "target": result.get("target"),
            "endpoint": result.get("endpoint"),
            "sent_at": now,
            "status_code": result.get("status_code"),
        }
    state["updated_at"] = now
    return state


def build_outreach_payload(target: OutreachTarget, *, canonical: str = "https://foresea.ink") -> Dict[str, Any]:
    """Build the JSON sent to one agent endpoint."""
    packet = build_pr_agent_packet(audience=target.audience, canonical=canonical)
    return {
        "type": "foresea.pr_agent.outreach",
        "from": "Foresea PR Agent",
        "to": target.name,
        "audience": packet["audience"],
        "subject": "Foresea: prediction-market intelligence for AI agents",
        "message": packet["message"],
        "packet": packet,
        "reply": {
            "mcp": packet["links"]["mcp"],
            "agent_manifest": packet["links"]["agent_manifest"],
            "pr_agent": f"{packet['links']['site']}/pr-agent",
        },
    }


def send_outreach(
    targets: Iterable[OutreachTarget],
    *,
    canonical: str = "https://foresea.ink",
    send: bool = False,
    session: Optional[Any] = None,
    timeout_s: float = 20.0,
    pause_s: float = 2.0,
) -> List[Dict[str, Any]]:
    """Prepare or send outreach to targets.

    When ``send`` is false, no network calls are made and each result includes
    the payload that would be sent.
    """
    if session is None and send:
        import requests

        session = requests.Session()
    results: List[Dict[str, Any]] = []
    target_list = list(targets)
    for i, target in enumerate(target_list):
        payload = build_outreach_payload(target, canonical=canonical)
        if not send:
            result = {
                "target": target.name,
                "endpoint": target.endpoint,
                "status": "dry_run",
                "transport": target.transport,
            }
            if target.transport == "form":
                result["form"] = target.form
            elif target.transport == "json":
                result["body"] = target.body
            else:
                result["payload"] = payload
            results.append(result)
            continue
        headers = {"User-Agent": "foresea-pr-agent/0.1"}
        if target.transport in {"webhook", "json"}:
            headers["Content-Type"] = "application/json"
        headers.update(target.headers or {})
        try:
            if target.transport == "form":
                response = session.post(target.endpoint, headers=headers, data=target.form, timeout=timeout_s)
            elif target.transport == "json":
                response = session.post(target.endpoint, headers=headers, json=target.body, timeout=timeout_s)
            else:
                response = session.post(target.endpoint, headers=headers, json=payload, timeout=timeout_s)
            ok = 200 <= int(response.status_code) < 300
            if ok and target.transport == "json":
                try:
                    parsed = response.json()
                    if isinstance(parsed, dict):
                        ok = not parsed.get("error") and int(parsed.get("code", 0)) == 0
                except Exception:  # noqa: BLE001
                    pass
            body = getattr(response, "text", "")[:500]
            results.append({
                "target": target.name,
                "endpoint": target.endpoint,
                "status": "sent" if ok else "failed",
                "status_code": response.status_code,
                "response": body,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "target": target.name,
                "endpoint": target.endpoint,
                "status": "failed",
                "error": str(exc),
            })
        if i < len(target_list) - 1 and pause_s > 0:
            time.sleep(pause_s)
    return results
