from __future__ import annotations

import fcntl
import json
import random
import hashlib
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from analyzing_llm_rationale.providers import (
    ChatProvider,
    ContextLimitError,
    ProviderResponseError,
    RetryableProviderError,
)
from analyzing_llm_rationale.validation import validate_dataset_records, validate_result_records

ARTICLE_FIELDS = (
    "url",
    "title",
    "authors",
    "publish_date",
    "summary",
    "summary_llm",
    "keywords",
    "frs",
    "credibility",
    "text",
)

SUMMARY_ARTICLE_FIELDS = (
    "title",
    "publish_date",
    "summary",
    "summary_llm",
    "keywords",
    "frs",
    "credibility",
)

DIGIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


@dataclass
class RunConfig:
    input_path: Path
    output_path: Path
    error_log_path: Path
    system_prompt_path: Path
    user_prompt_path: Path
    output_fields: Sequence[str]
    temperature: float = 0.0
    max_tokens: int = 2048
    max_records: int = 0
    max_attempts: int = 3
    retry_base_sleep_s: float = 1.5
    reprocess_null_only: bool = False
    drop_article_text: bool = False
    variant_name: str = ""
    model_key: str = ""
    model_label: str = ""
    provider_name: str = ""
    model_identifier: str = ""
    temperature_tag: str = ""
    run_metadata_path: Optional[Path] = None
    shard_count: int = 1
    shard_index: int = 0
    progress_every: int = 0


@dataclass
class RunSummary:
    processed: int
    total_results: int
    null_predictions: int
    output_path: Path


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def log_error(error_log_path: Path, event: Dict[str, object]) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with error_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False))
        handle.write("\n")


def extract_summary_items(record: Dict[str, object], article_detail: str) -> List[Dict[str, object]]:
    summary_items: List[Dict[str, object]] = []
    for article in record.get("news_articles") or []:
        if not isinstance(article, dict):
            continue
        item = {}
        if article_detail == "summary":
            for field in SUMMARY_ARTICLE_FIELDS:
                item[field] = article.get(field)
            if not item.get("summary_llm") and item.get("summary"):
                item["summary_llm"] = item.get("summary")
        else:
            for field in ARTICLE_FIELDS:
                item[field] = article.get(field)
        summary_items.append(item)
    return summary_items


def build_user_prompt(
    record: Dict[str, object],
    user_prompt_template: str,
    article_detail: str,
) -> str:
    question = str(record.get("question") or "").strip()
    description = str(record.get("description") or "").strip()
    resolution = str(record.get("resolution_criteria") or "").strip()
    categories = record.get("categories") or []
    created_time = record.get("created_time")
    publish_time = record.get("publish_time")
    resolve_time = record.get("resolve_time")
    days_open = record.get("days_open")

    prompt_suffix = user_prompt_template.replace("[question]", "").strip()
    parts = [f"Question: {question}"]
    if description:
        parts.append(f"Description: {description}")
    if resolution:
        parts.append(f"Resolution Criteria: {resolution}")
    if categories:
        parts.append(f"Categories: {categories}")
    if created_time:
        parts.append(f"Created Time: {created_time}")
    if publish_time:
        parts.append(f"Publish Time: {publish_time}")
    if resolve_time:
        parts.append(f"Resolve Time: {resolve_time}")
    if days_open is not None:
        parts.append(f"Days Open: {days_open}")

    if article_detail == "summary":
        parts.append("Evidence Summaries:")
    else:
        parts.append("Evidence Summaries (full article fields):")
    summary_items = extract_summary_items(record, article_detail=article_detail)
    if summary_items:
        for index, item in enumerate(summary_items, start=1):
            parts.append(f"Article {index}: {json.dumps(item, ensure_ascii=False)}")
    else:
        parts.append("(none)")

    parts.append("")
    parts.append(prompt_suffix)
    return "\n".join(parts).strip()


def _parse_json_dict(value: object) -> Optional[Dict[str, object]]:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidates.append("\n".join(lines[1:-1]).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1].strip())

    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_from_jsonish_text(value: object, output_fields: Sequence[str]) -> Dict[str, object]:
    extracted = {field: None for field in output_fields}
    if not isinstance(value, str):
        return extracted

    for field in output_fields:
        if field == "confidence":
            extracted[field] = _extract_confidence_from_text(value)
            continue

        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            value,
            flags=re.S,
        )
        if match:
            try:
                extracted[field] = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                extracted[field] = match.group(1)
        elif field == "rationale":
            # Handle truncated or malformed JSON strings where the closing quote is missing.
            loose_match = re.search(
                rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)(?:\"|$)',
                value,
                flags=re.S,
            )
            if loose_match:
                try:
                    extracted[field] = json.loads(f'"{loose_match.group(1)}"')
                except json.JSONDecodeError:
                    extracted[field] = loose_match.group(1)

    return extracted


def _coerce_predicted_answer(value: object) -> object:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if not isinstance(value, str):
        return value

    normalized = value.strip().lower()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
        return "No"
    return value.strip()


def _coerce_string_field(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _coerce_confidence(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text:
        return None

    if text.endswith("%"):
        try:
            return float(text[:-1].strip()) / 100.0
        except ValueError:
            return None

    try:
        return float(text)
    except ValueError:
        pass

    # Handle malformed forms such as `0. nine` or `0.nine`.
    match = re.fullmatch(r"(-?\d+)\s*\.\s*([a-z]+)", text)
    if match and match.group(2) in DIGIT_WORDS:
        return float(f"{match.group(1)}.{DIGIT_WORDS[match.group(2)]}")

    if text in {"zero", "one"}:
        return float(DIGIT_WORDS[text])

    return None


def _extract_confidence_from_text(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None

    patterns = (
        r'"confidence"\s*:\s*(-?\d+)\s*\.\s*([A-Za-z]+)',
        r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)',
        r'"confidence"\s*:\s*"([^"]+)"',
    )

    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        if len(match.groups()) == 1:
            coerced = _coerce_confidence(match.group(1))
        else:
            coerced = _coerce_confidence(f"{match.group(1)}.{match.group(2)}")
        if coerced is not None:
            return coerced
    return None


def _normalize_result_fields(
    result: Dict[str, object],
    output_fields: Sequence[str],
    raw_content: Optional[str] = None,
) -> Dict[str, object]:
    normalized = dict(result)

    if "predicted_answer" in normalized:
        normalized["predicted_answer"] = _coerce_predicted_answer(normalized.get("predicted_answer"))

    if "confidence" in output_fields:
        confidence = _coerce_confidence(normalized.get("confidence"))
        if confidence is None and raw_content is not None:
            confidence = _extract_confidence_from_text(raw_content)
        normalized["confidence"] = confidence

    for field in output_fields:
        if field in {"predicted_answer", "confidence"}:
            continue
        normalized[field] = _coerce_string_field(normalized.get(field))

    return normalized


def recover_missing_fields(result: Dict[str, object], output_fields: Sequence[str]) -> Dict[str, object]:
    recovered = dict(result)
    nested = _parse_json_dict(recovered.get("rationale"))
    if nested is None:
        nested = _extract_from_jsonish_text(recovered.get("rationale"), output_fields)
        if not any(nested.get(field) is not None for field in output_fields):
            return recovered

    for field in output_fields:
        if recovered.get(field) is None and nested.get(field) is not None:
            recovered[field] = nested.get(field)

    nested_rationale = nested.get("rationale")
    if isinstance(nested_rationale, str):
        recovered["rationale"] = nested_rationale
    return _normalize_result_fields(recovered, output_fields, raw_content=recovered.get("rationale"))


def parse_model_response(content: str, output_fields: Sequence[str]) -> Dict[str, object]:
    default_result = {field: None for field in output_fields}
    parsed = _parse_json_dict(content)
    if parsed is None:
        if "rationale" in default_result:
            default_result["rationale"] = content
        return recover_missing_fields(default_result, output_fields)

    for field in output_fields:
        default_result[field] = parsed.get(field)
    normalized = _normalize_result_fields(default_result, output_fields, raw_content=content)
    return recover_missing_fields(normalized, output_fields)


def load_existing_results(output_path: Path) -> List[Dict[str, object]]:
    if not output_path.exists():
        return []
    existing = load_json(output_path)
    return existing if isinstance(existing, list) else []


def merge_result_row_locked(
    output_path: Path,
    records: Sequence[Dict[str, object]],
    result_row: Dict[str, object],
) -> List[Dict[str, object]]:
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        latest_results = load_existing_results(output_path)
        latest_by_id = {
            row.get("id"): row
            for row in latest_results
            if isinstance(row, dict) and row.get("id") is not None
        }
        latest_by_id[result_row.get("id")] = result_row
        ordered_results = _ordered_results(records, latest_by_id)
        write_json_atomic(output_path, ordered_results)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return ordered_results


def count_null_predictions(results: Iterable[Dict[str, object]]) -> int:
    return sum(1 for row in results if row.get("predicted_answer") is None)


def is_effectively_empty_result(result: Dict[str, object]) -> bool:
    return all(value is None for value in result.values())


def pending_record_ids(
    records: Sequence[Dict[str, object]],
    existing_results: Sequence[Dict[str, object]],
    reprocess_null_only: bool,
) -> List[object]:
    if reprocess_null_only:
        return [
            row.get("id")
            for row in existing_results
            if isinstance(row, dict) and row.get("predicted_answer") is None
        ]

    completed_ids = {
        row.get("id")
        for row in existing_results
        if isinstance(row, dict) and row.get("id") is not None
    }
    return [
        record.get("id")
        for record in records
        if isinstance(record, dict) and record.get("id") not in completed_ids
    ]


def filter_shard_ids(
    record_ids: Sequence[object],
    shard_count: int,
    shard_index: int,
) -> List[object]:
    if shard_count <= 1:
        return list(record_ids)
    filtered: List[object] = []
    for rid in record_ids:
        if isinstance(rid, int) and rid % shard_count == shard_index:
            filtered.append(rid)
    return filtered


def compute_sleep_s(base_sleep_s: float, attempt: int) -> float:
    bounded = min(120.0, base_sleep_s * (2 ** min(attempt, 6)))
    return bounded * (0.75 + 0.5 * random.random())


def looks_like_timeout_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "timed out",
        "timeout",
        "remote end closed connection",
        "connection aborted",
    )
    return any(marker in text for marker in markers)


def effective_max_tokens_for_attempt(
    base_max_tokens: int,
    attempt: int,
    article_detail: str,
) -> int:
    reduced = base_max_tokens
    if article_detail != "full":
        reduced = min(reduced, 1024)
    if attempt >= 1:
        reduced = min(reduced, 1024)
    if attempt >= 2:
        reduced = min(reduced, 768)
    if attempt >= 3:
        reduced = min(reduced, 512)
    if attempt >= 4:
        reduced = min(reduced, 384)
    return max(256, reduced)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_run_metadata(
    config: RunConfig,
    system_prompt: str,
    user_prompt_template: str,
    input_record_count: int,
    existing_result_count: int,
    status: str,
    summary: Optional[RunSummary] = None,
) -> Dict[str, object]:
    metadata = {
        "status": status,
        "variant": config.variant_name,
        "model_key": config.model_key,
        "model_label": config.model_label,
        "provider": config.provider_name,
        "model_identifier": config.model_identifier,
        "temperature": config.temperature,
        "temperature_tag": config.temperature_tag,
        "max_tokens": config.max_tokens,
        "max_attempts": config.max_attempts,
        "retry_base_sleep_s": config.retry_base_sleep_s,
        "reprocess_null_only": config.reprocess_null_only,
        "drop_article_text": config.drop_article_text,
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "error_log_path": str(config.error_log_path),
        "system_prompt_path": str(config.system_prompt_path),
        "user_prompt_path": str(config.user_prompt_path),
        "output_fields": list(config.output_fields),
        "input_record_count": input_record_count,
        "existing_result_count": existing_result_count,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt_template),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if summary is not None:
        metadata["summary"] = {
            "processed": summary.processed,
            "total_results": summary.total_results,
            "null_predictions": summary.null_predictions,
        }
    return metadata


def process_batch(config: RunConfig, provider: ChatProvider) -> RunSummary:
    records = load_json(config.input_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {config.input_path}")
    validate_dataset_records(records)
    record_by_id = {
        record.get("id"): record
        for record in records
        if isinstance(record, dict) and record.get("id") is not None
    }

    existing_results = load_existing_results(config.output_path)
    validate_result_records(existing_results, config.output_fields, allow_partial=True)
    results_by_id = {
        row.get("id"): row
        for row in existing_results
        if isinstance(row, dict) and row.get("id") is not None
    }

    system_prompt = config.system_prompt_path.read_text(encoding="utf-8").strip()
    user_prompt_template = config.user_prompt_path.read_text(encoding="utf-8").strip()
    if config.run_metadata_path is not None:
        write_json(
            config.run_metadata_path,
            build_run_metadata(
                config=config,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                input_record_count=len(records),
                existing_result_count=len(existing_results),
                status="running",
            ),
        )

    todo_ids = pending_record_ids(records, existing_results, config.reprocess_null_only)
    if config.shard_count > 1:
        todo_ids = filter_shard_ids(todo_ids, config.shard_count, config.shard_index)
    processed = 0

    total_todo = len(todo_ids)

    for record_id in todo_ids:
        record = record_by_id.get(record_id)
        if record is None:
            continue
        if config.max_records and processed >= config.max_records:
            break

        article_detail = "summary" if config.drop_article_text else "full"
        content = None
        last_exception: Optional[Exception] = None

        for attempt in range(config.max_attempts):
            attempt_max_tokens = effective_max_tokens_for_attempt(
                config.max_tokens,
                attempt,
                article_detail=article_detail,
            )
            user_prompt = build_user_prompt(
                record,
                user_prompt_template=user_prompt_template,
                article_detail=article_detail,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                content = provider.chat_completion(
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=attempt_max_tokens,
                )
                break
            except ContextLimitError as exc:
                last_exception = exc
                if article_detail == "full":
                    article_detail = "summary"
                    log_error(
                        config.error_log_path,
                        {
                            "id": record_id,
                            "phase": "context_trim",
                            "attempt": attempt,
                            "detail": str(exc),
                        },
                    )
                    continue
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "context_limit",
                        "attempt": attempt,
                        "detail": str(exc),
                    },
                )
                break
            except RetryableProviderError as exc:
                last_exception = exc
                sleep_s = compute_sleep_s(config.retry_base_sleep_s, attempt)
                if article_detail == "full":
                    article_detail = "summary"
                    log_error(
                        config.error_log_path,
                        {
                            "id": record_id,
                            "phase": "latency_trim",
                            "attempt": attempt,
                            "detail": "Switching to summary-only evidence after retryable provider failure.",
                        },
                    )
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "provider_retry",
                        "attempt": attempt,
                        "detail": str(exc),
                        "sleep_s": round(sleep_s, 3),
                    },
                )
                if attempt < config.max_attempts - 1:
                    time.sleep(sleep_s)
                    continue
                break
            except ProviderResponseError as exc:
                last_exception = exc
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "provider_fail",
                        "attempt": attempt,
                        "detail": str(exc),
                    },
                )
                break
            except Exception as exc:
                last_exception = exc
                sleep_s = compute_sleep_s(config.retry_base_sleep_s, attempt)
                if article_detail == "full" and looks_like_timeout_error(exc):
                    article_detail = "summary"
                    log_error(
                        config.error_log_path,
                        {
                            "id": record_id,
                            "phase": "latency_trim",
                            "attempt": attempt,
                            "detail": "Switching to summary-only evidence after timeout-like exception.",
                        },
                    )
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "exception",
                        "attempt": attempt,
                        "exc": repr(exc),
                        "trace": traceback.format_exc(limit=5),
                        "sleep_s": round(sleep_s, 3),
                    },
                )
                if attempt < config.max_attempts - 1:
                    time.sleep(sleep_s)
                    continue
                break

        parsed_result = {field: None for field in config.output_fields}
        should_write_result = False
        if content is not None:
            parsed_result = parse_model_response(content, config.output_fields)
            if is_effectively_empty_result(parsed_result):
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "parse_json",
                        "content_head": content[:500],
                    },
                )
            else:
                should_write_result = True
        elif last_exception is not None:
            log_error(
                config.error_log_path,
                {
                    "id": record_id,
                    "phase": "record_failed",
                    "detail": str(last_exception),
                },
            )

        if should_write_result:
            result_row = {"id": record_id, **parsed_result}
            results_by_id[record_id] = result_row
            ordered_results = merge_result_row_locked(config.output_path, records, result_row)
            results_by_id = {
                row.get("id"): row
                for row in ordered_results
                if isinstance(row, dict) and row.get("id") is not None
            }
        processed += 1
        if config.progress_every > 0 and (
            processed == 1 or processed % config.progress_every == 0 or processed == total_todo
        ):
            print(
                f"PROGRESS record_id={record_id} processed={processed}/{total_todo} "
                f"wrote={'yes' if should_write_result else 'no'} "
                f"results={len(results_by_id)} nulls={count_null_predictions(results_by_id.values())}",
                flush=True,
            )

    final_results = _ordered_results(records, results_by_id)
    summary = RunSummary(
        processed=processed,
        total_results=len(final_results),
        null_predictions=count_null_predictions(final_results),
        output_path=config.output_path,
    )
    if config.run_metadata_path is not None:
        write_json(
            config.run_metadata_path,
            build_run_metadata(
                config=config,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                input_record_count=len(records),
                existing_result_count=len(existing_results),
                status="completed",
                summary=summary,
            ),
        )
    return summary


def _ordered_results(
    records: Sequence[Dict[str, object]],
    results_by_id: Dict[object, Dict[str, object]],
) -> List[Dict[str, object]]:
    ordered: List[Dict[str, object]] = []
    seen = set()
    for record in records:
        record_id = record.get("id")
        if record_id in results_by_id:
            ordered.append(results_by_id[record_id])
            seen.add(record_id)
    for record_id, result in results_by_id.items():
        if record_id not in seen:
            ordered.append(result)
    return ordered
