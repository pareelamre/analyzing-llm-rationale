from __future__ import annotations

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


def log_error(error_log_path: Path, event: Dict[str, object]) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event)
    event["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with error_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False))
        handle.write("\n")


def extract_summary_items(record: Dict[str, object], include_article_text: bool) -> List[Dict[str, object]]:
    summary_items: List[Dict[str, object]] = []
    for article in record.get("news_articles") or []:
        if not isinstance(article, dict):
            continue
        item = {}
        for field in ARTICLE_FIELDS:
            if field == "text" and not include_article_text:
                item[field] = None
            else:
                item[field] = article.get(field)
        summary_items.append(item)
    return summary_items


def build_user_prompt(
    record: Dict[str, object],
    user_prompt_template: str,
    include_article_text: bool,
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

    parts.append("Evidence Summaries (full article fields):")
    summary_items = extract_summary_items(record, include_article_text=include_article_text)
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
            match = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', value)
            if match:
                extracted[field] = float(match.group(1))
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

    return extracted


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
    return recovered


def parse_model_response(content: str, output_fields: Sequence[str]) -> Dict[str, object]:
    default_result = {field: None for field in output_fields}
    parsed = _parse_json_dict(content)
    if parsed is None:
        if "rationale" in default_result:
            default_result["rationale"] = content
        return default_result

    for field in output_fields:
        default_result[field] = parsed.get(field)
    return recover_missing_fields(default_result, output_fields)


def load_existing_results(output_path: Path) -> List[Dict[str, object]]:
    if not output_path.exists():
        return []
    existing = load_json(output_path)
    return existing if isinstance(existing, list) else []


def count_null_predictions(results: Iterable[Dict[str, object]]) -> int:
    return sum(1 for row in results if row.get("predicted_answer") is None)


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


def compute_sleep_s(base_sleep_s: float, attempt: int) -> float:
    bounded = min(120.0, base_sleep_s * (2 ** min(attempt, 6)))
    return bounded * (0.75 + 0.5 * random.random())


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
    processed = 0

    for record_id in todo_ids:
        record = record_by_id.get(record_id)
        if record is None:
            continue
        if config.max_records and processed >= config.max_records:
            break

        include_article_text = not config.drop_article_text
        content = None
        last_exception: Optional[Exception] = None

        for attempt in range(config.max_attempts):
            user_prompt = build_user_prompt(
                record,
                user_prompt_template=user_prompt_template,
                include_article_text=include_article_text,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                content = provider.chat_completion(
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                break
            except ContextLimitError as exc:
                last_exception = exc
                if include_article_text:
                    include_article_text = False
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
        if content is not None:
            parsed_result = parse_model_response(content, config.output_fields)
            if all(value is None for value in parsed_result.values()):
                log_error(
                    config.error_log_path,
                    {
                        "id": record_id,
                        "phase": "parse_json",
                        "content_head": content[:500],
                    },
                )
        elif last_exception is not None:
            log_error(
                config.error_log_path,
                {
                    "id": record_id,
                    "phase": "record_failed",
                    "detail": str(last_exception),
                },
            )

        result_row = {"id": record_id, **parsed_result}
        results_by_id[record_id] = result_row
        ordered_results = _ordered_results(records, results_by_id)
        write_json(config.output_path, ordered_results)
        processed += 1

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
