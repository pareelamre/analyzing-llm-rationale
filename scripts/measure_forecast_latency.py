"""Measure Foresea forecast latency with reproducible HTTP requests.

Examples:
  py scripts/measure_forecast_latency.py --url https://foresea.ink --models default,minimax-m3 --runs 2
  py scripts/measure_forecast_latency.py --url http://127.0.0.1:8080 --mode stream --bearer-token "$TOKEN"
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

DEFAULT_QUESTION = "Will the Federal Reserve cut rates before December 31, 2026?"


def _headers(args: argparse.Namespace) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if args.api_key:
        headers["x-api-key"] = args.api_key
    if args.bearer_token:
        headers["authorization"] = f"Bearer {args.bearer_token}"
    return headers


def _payload(args: argparse.Namespace, model: str, run_index: int) -> dict[str, Any]:
    question = args.question
    if args.cache_bust:
        question = f"{question} Measurement nonce {run_index}-{str(uuid.uuid4())[:8]}."
    payload: dict[str, Any] = {
        "question": question,
        "variant": args.variant,
        "chat_mode": args.chat_mode,
        "attach_evidence": args.attach_evidence,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    if args.evidence_top_k is not None:
        payload["evidence_top_k"] = args.evidence_top_k
    if model != "default":
        payload["model"] = model
    return payload


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            elapsed = time.perf_counter() - started
            parsed = json.loads(body.decode("utf-8")) if body else {}
            return {
                "status": resp.status,
                "seconds": round(elapsed, 3),
                "bytes": len(body),
                "model_key": parsed.get("model_key"),
                "served_model_name": parsed.get("served_model_name"),
                "evidence_error": parsed.get("evidence_error"),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": exc.code,
            "seconds": round(time.perf_counter() - started, 3),
            "error_body": body[:500],
        }
    except Exception as exc:  # pragma: no cover - network/environment dependent.
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "error": type(exc).__name__,
            "detail": str(exc)[:500],
        }


def _sse_event(event: str | None, data_lines: list[str]) -> tuple[str | None, dict[str, Any]]:
    if event is None:
        return None, {}
    data = "\n".join(data_lines).strip()
    if not data:
        return event, {}
    try:
        return event, json.loads(data)
    except json.JSONDecodeError:
        return event, {"raw": data}


def _post_stream(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    result: dict[str, Any] = {
        "status": None,
        "meta_streaming_ms": None,
        "first_delta_ms": None,
        "done_ms": None,
        "server_prepare_ms": None,
        "server_first_delta_ms": None,
        "provider_first_delta_ms": None,
        "cache_hit": None,
        "delta_chars": 0,
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status"] = resp.status
            event: str | None = None
            data_lines: list[str] = []
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("event: "):
                    event = line[7:]
                    data_lines = []
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line == "" and event:
                    name, data = _sse_event(event, data_lines)
                    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    if name == "meta" and data.get("status") == "streaming":
                        result["meta_streaming_ms"] = elapsed_ms
                        result["server_prepare_ms"] = data.get("prepare_ms")
                        result["model_key"] = data.get("model_key")
                        result["evidence_error"] = data.get("evidence_error")
                        result["cache_hit"] = data.get("cache_hit", False)
                    elif name == "delta":
                        text = data.get("text") or ""
                        result["delta_chars"] += len(text)
                        if result["first_delta_ms"] is None:
                            result["first_delta_ms"] = elapsed_ms
                            result["server_first_delta_ms"] = data.get("first_delta_ms")
                            result["provider_first_delta_ms"] = data.get("provider_first_delta_ms")
                            if "cache_hit" in data:
                                result["cache_hit"] = data.get("cache_hit")
                    elif name == "done":
                        result["done_ms"] = elapsed_ms
                        response = data.get("response") if isinstance(data, dict) else None
                        if isinstance(response, dict):
                            result["response_model_key"] = response.get("model_key")
                            result["served_model_name"] = response.get("served_model_name")
                            if result.get("evidence_error") is None:
                                result["evidence_error"] = response.get("evidence_error")
                        break
                    elif name == "error":
                        result["error_event"] = data
                        if isinstance(data, dict) and data.get("status_code") is not None:
                            result["status"] = data.get("status_code")
                        break
                    event = None
                    data_lines = []
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        result.update({
            "status": exc.code,
            "seconds": round(time.perf_counter() - started, 3),
            "error_body": body[:500],
        })
        return result
    except Exception as exc:  # pragma: no cover - network/environment dependent.
        result.update({
            "seconds": round(time.perf_counter() - started, 3),
            "error": type(exc).__name__,
            "detail": str(exc)[:500],
        })
        return result


def _summarize(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    metric = "done_ms" if mode == "stream" else "seconds"
    out: list[dict[str, Any]] = []
    for model in sorted({str(row.get("model")) for row in rows}):
        model_rows = [row for row in rows if row.get("model") == model and row.get("status") == 200]
        values = [float(row[metric]) for row in model_rows if row.get(metric) is not None]
        first_delta = [
            float(row["first_delta_ms"])
            for row in model_rows
            if row.get("first_delta_ms") is not None
        ]
        server_prepare = [
            float(row["server_prepare_ms"])
            for row in model_rows
            if row.get("server_prepare_ms") is not None
        ]
        server_first_delta = [
            float(row["server_first_delta_ms"])
            for row in model_rows
            if row.get("server_first_delta_ms") is not None
        ]
        provider_first_delta = [
            float(row["provider_first_delta_ms"])
            for row in model_rows
            if row.get("provider_first_delta_ms") is not None
        ]
        summary: dict[str, Any] = {"model": model, "successes": len(model_rows)}
        if values:
            summary.update({
                f"avg_{metric}": round(statistics.mean(values), 3),
                f"min_{metric}": round(min(values), 3),
                f"max_{metric}": round(max(values), 3),
            })
        if first_delta:
            summary.update({
                "avg_first_delta_ms": round(statistics.mean(first_delta), 3),
                "min_first_delta_ms": round(min(first_delta), 3),
                "max_first_delta_ms": round(max(first_delta), 3),
            })
        if server_prepare:
            summary["avg_server_prepare_ms"] = round(statistics.mean(server_prepare), 3)
        if server_first_delta:
            summary["avg_server_first_delta_ms"] = round(statistics.mean(server_first_delta), 3)
        if provider_first_delta:
            summary["avg_provider_first_delta_ms"] = round(statistics.mean(provider_first_delta), 3)
        out.append(summary)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Foresea /predict latency.")
    parser.add_argument("--url", default="https://foresea.ink", help="Base URL, e.g. https://foresea.ink")
    parser.add_argument("--mode", choices=("blocking", "stream"), default="blocking")
    parser.add_argument("--models", default="default", help="Comma-separated model labels; use default for server default")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--variant", default="variant0_neutral_baseline")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--evidence-top-k", type=int, default=None)
    parser.add_argument("--attach-evidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chat-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-bust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key value")
    parser.add_argument("--bearer-token", default=None, help="Optional bearer token for authenticated routes")
    args = parser.parse_args()

    endpoint = "/predict/stream" if args.mode == "stream" else "/predict"
    url = args.url.rstrip("/") + endpoint
    headers = _headers(args)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    rows: list[dict[str, Any]] = []

    for model in models:
        for run_index in range(1, max(1, args.runs) + 1):
            payload = _payload(args, model, run_index)
            if args.mode == "stream":
                measured = _post_stream(url, payload, headers, args.timeout)
            else:
                measured = _post_json(url, payload, headers, args.timeout)
            measured.update({
                "model": model,
                "run": run_index,
                "attach_evidence": args.attach_evidence,
                "max_tokens": args.max_tokens,
                "cache_bust": args.cache_bust,
            })
            rows.append(measured)

    print(json.dumps({
        "url": args.url.rstrip("/"),
        "mode": args.mode,
        "rows": rows,
        "summary": _summarize(rows, args.mode),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
