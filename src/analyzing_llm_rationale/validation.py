from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence


class SchemaValidationError(ValueError):
    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("\n".join(self.issues))


@dataclass
class VerificationSummary:
    total_rows: int
    complete_rows: int
    malformed_rows: List[int]
    duplicate_ids: List[object]
    incomplete_ids: List[object]
    null_prediction_ids: List[object]
    missing_result_ids: List[object]
    unexpected_result_ids: List[object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def is_clean(self) -> bool:
        return not (
            self.malformed_rows
            or self.duplicate_ids
            or self.incomplete_ids
            or self.missing_result_ids
            or self.unexpected_result_ids
        )


def _append_issue(issues: List[str], path: str, message: str) -> None:
    issues.append(f"{path}: {message}")


def _normalize_binary_answer(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"yes", "no"}:
        return normalized
    return None


def _normalize_confidence(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence) or math.isinf(confidence):
        return None
    if not (0.0 <= confidence <= 1.0):
        return None
    return confidence


def is_incomplete_prediction_row(row: object) -> bool:
    if not isinstance(row, dict):
        return True
    if _normalize_binary_answer(row.get("predicted_answer")) is None:
        return True
    if _normalize_confidence(row.get("confidence")) is None:
        return True
    return False


def validate_dataset_records(records: object) -> None:
    issues: List[str] = []
    if not isinstance(records, list):
        raise SchemaValidationError(["dataset: expected a top-level list"])

    seen_ids = set()
    for index, record in enumerate(records):
        path = f"dataset[{index}]"
        if not isinstance(record, dict):
            _append_issue(issues, path, "expected object")
            continue
        record_id = record.get("id")
        if record_id is None:
            _append_issue(issues, path, "missing id")
        elif record_id in seen_ids:
            _append_issue(issues, path, f"duplicate id {record_id!r}")
        else:
            seen_ids.add(record_id)

        for field in ("question", "description", "resolution_criteria"):
            value = record.get(field)
            if value is not None and not isinstance(value, str):
                _append_issue(issues, path, f"{field} must be a string when present")

        categories = record.get("categories")
        if categories is not None and not isinstance(categories, list):
            _append_issue(issues, path, "categories must be a list when present")

        articles = record.get("news_articles")
        if articles is not None and not isinstance(articles, list):
            _append_issue(issues, path, "news_articles must be a list when present")
            continue
        if isinstance(articles, list):
            for article_index, article in enumerate(articles):
                article_path = f"{path}.news_articles[{article_index}]"
                if not isinstance(article, dict):
                    _append_issue(issues, article_path, "expected object")
                    continue
                for field in ("url", "title", "summary", "summary_llm", "text"):
                    value = article.get(field)
                    if value is not None and not isinstance(value, str):
                        _append_issue(issues, article_path, f"{field} must be a string when present")

    if issues:
        raise SchemaValidationError(issues)


def validate_result_records(
    records: object,
    expected_fields: Sequence[str],
    allow_partial: bool,
) -> None:
    issues: List[str] = []
    if not isinstance(records, list):
        raise SchemaValidationError(["results: expected a top-level list"])

    seen_ids = set()
    for index, row in enumerate(records):
        path = f"results[{index}]"
        if not isinstance(row, dict):
            _append_issue(issues, path, "expected object")
            continue
        row_id = row.get("id")
        if row_id is None:
            _append_issue(issues, path, "missing id")
            continue
        if row_id in seen_ids:
            _append_issue(issues, path, f"duplicate id {row_id!r}")
        else:
            seen_ids.add(row_id)

        if not allow_partial:
            missing = [field for field in expected_fields if field not in row]
            if missing:
                _append_issue(issues, path, f"missing expected fields: {', '.join(missing)}")

    if issues:
        raise SchemaValidationError(issues)


def verify_result_records(
    records: object,
    expected_fields: Sequence[str],
    dataset_records: Optional[object] = None,
) -> VerificationSummary:
    if not isinstance(records, list):
        raise SchemaValidationError(["results: expected a top-level list"])

    malformed_rows: List[int] = []
    duplicate_ids: List[object] = []
    incomplete_ids: List[object] = []
    null_prediction_ids: List[object] = []
    seen_ids = set()
    result_ids = []
    complete_rows = 0

    for index, row in enumerate(records):
        if not isinstance(row, dict):
            malformed_rows.append(index)
            continue
        row_id = row.get("id")
        if row_id is None:
            malformed_rows.append(index)
            continue
        result_ids.append(row_id)
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        else:
            seen_ids.add(row_id)

        missing_fields = [field for field in expected_fields if field not in row]
        if missing_fields:
            incomplete_ids.append(row_id)
            continue
        if is_incomplete_prediction_row(row):
            null_prediction_ids.append(row_id)
            incomplete_ids.append(row_id)
            continue
        complete_rows += 1

    missing_result_ids: List[object] = []
    unexpected_result_ids: List[object] = []
    if dataset_records is not None:
        validate_dataset_records(dataset_records)
        assert isinstance(dataset_records, list)
        dataset_ids = [
            record.get("id")
            for record in dataset_records
            if isinstance(record, dict) and record.get("id") is not None
        ]
        dataset_id_set = set(dataset_ids)
        result_id_set = set(result_ids)
        missing_result_ids = sorted(dataset_id_set - result_id_set)
        unexpected_result_ids = sorted(result_id_set - dataset_id_set)

    return VerificationSummary(
        total_rows=len(records),
        complete_rows=complete_rows,
        malformed_rows=malformed_rows,
        duplicate_ids=duplicate_ids,
        incomplete_ids=incomplete_ids,
        null_prediction_ids=null_prediction_ids,
        missing_result_ids=missing_result_ids,
        unexpected_result_ids=unexpected_result_ids,
    )
