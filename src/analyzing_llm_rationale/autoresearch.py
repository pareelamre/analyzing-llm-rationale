from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from analyzing_llm_rationale.metrics import accuracy, brier_score, ece, iter_examples, load_targets
from analyzing_llm_rationale.pipeline import RunConfig, process_batch, sha256_text

MetricName = Literal["brier_score", "accuracy", "ece"]


class ForecastProvider(Protocol):
    def chat_completion(self, messages, temperature: float, max_tokens: int) -> str:
        ...


@dataclass(frozen=True)
class AutoresearchConfig:
    input_path: Path
    system_prompt_path: Path
    candidate_prompt_path: Path
    output_dir: Path
    baseline_results_path: Optional[Path] = None
    promote_to: Optional[Path] = None
    metric: MetricName = "brier_score"
    max_records: int = 50
    temperature: float = 0.0
    max_tokens: int = 1024
    max_attempts: int = 2
    retry_base_sleep_s: float = 1.5
    drop_article_text: bool = True
    bins: int = 10
    min_delta: float = 0.0
    model_key: str = ""
    model_label: str = ""
    provider_name: str = ""
    model_identifier: str = ""


@dataclass(frozen=True)
class Score:
    n_scored: int
    n_missing: int
    accuracy: Optional[float]
    brier_score: Optional[float]
    ece: Optional[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_scored": self.n_scored,
            "n_missing": self.n_missing,
            "accuracy": self.accuracy,
            "brier_score": self.brier_score,
            "ece": self.ece,
        }


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def score_results(results_path: Path, dataset_path: Path, *, bins: int = 10) -> Score:
    rows = _read_json_list(results_path)
    targets = load_targets(dataset_path)
    examples, missing = iter_examples(rows, targets)
    if not examples:
        return Score(
            n_scored=0,
            n_missing=missing,
            accuracy=None,
            brier_score=None,
            ece=None,
        )
    return Score(
        n_scored=len(examples),
        n_missing=missing,
        accuracy=accuracy(examples),
        brier_score=brier_score(examples),
        ece=ece(examples, bins=bins),
    )


def _metric_value(score: Score, metric: MetricName) -> Optional[float]:
    return getattr(score, metric)


def is_improvement(
    candidate: Score,
    baseline: Optional[Score],
    *,
    metric: MetricName,
    min_delta: float = 0.0,
) -> Optional[bool]:
    """Whether candidate improves baseline for the selected metric.

    Lower is better for Brier score and ECE; higher is better for accuracy.
    Returns None when no baseline is available or either score lacks the metric.
    """
    if baseline is None:
        return None
    candidate_value = _metric_value(candidate, metric)
    baseline_value = _metric_value(baseline, metric)
    if candidate_value is None or baseline_value is None:
        return None
    if metric == "accuracy":
        return candidate_value >= baseline_value + min_delta
    return candidate_value <= baseline_value - min_delta


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def run_autoresearch_once(config: AutoresearchConfig, provider: ForecastProvider) -> dict[str, object]:
    """Run one Foresea prompt experiment and log a comparable score.

    This intentionally runs one candidate prompt per invocation. An external agent
    can edit ``candidate_prompt_path`` and call this command in a loop, while this
    function keeps the measurement, promotion, and audit log deterministic.
    """
    prompt_text = config.candidate_prompt_path.read_text(encoding="utf-8").strip()
    if "[question]" not in prompt_text:
        raise ValueError("Candidate prompt must contain the [question] placeholder.")

    prompt_hash = sha256_text(prompt_text)[:12]
    run_id = f"{_utc_run_id()}_{prompt_hash}"
    run_dir = config.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prompt_snapshot = run_dir / "candidate_prompt.txt"
    prompt_snapshot.write_text(prompt_text + "\n", encoding="utf-8")

    output_path = run_dir / "results_autoresearch_candidate.json"
    error_log_path = run_dir / "errors_autoresearch_candidate.jsonl"
    metadata_path = run_dir / "run_metadata_autoresearch_candidate.json"
    run_config = RunConfig(
        input_path=config.input_path,
        output_path=output_path,
        error_log_path=error_log_path,
        system_prompt_path=config.system_prompt_path,
        user_prompt_path=prompt_snapshot,
        output_fields=("predicted_answer", "confidence", "rationale"),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_records=config.max_records,
        max_attempts=config.max_attempts,
        retry_base_sleep_s=config.retry_base_sleep_s,
        drop_article_text=config.drop_article_text,
        variant_name="autoresearch_candidate",
        model_key=config.model_key,
        model_label=config.model_label,
        provider_name=config.provider_name,
        model_identifier=config.model_identifier,
        temperature_tag=f"temperature_{config.temperature:g}",
        run_metadata_path=metadata_path,
    )

    summary = process_batch(run_config, provider)
    candidate_score = score_results(output_path, config.input_path, bins=config.bins)
    baseline_score = (
        score_results(config.baseline_results_path, config.input_path, bins=config.bins)
        if config.baseline_results_path
        else None
    )
    improved = is_improvement(
        candidate_score,
        baseline_score,
        metric=config.metric,
        min_delta=config.min_delta,
    )

    promoted_to = None
    if improved is True and config.promote_to is not None:
        config.promote_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prompt_snapshot, config.promote_to)
        promoted_to = str(config.promote_to)

    report: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prompt_hash": prompt_hash,
        "candidate_prompt_path": str(config.candidate_prompt_path),
        "prompt_snapshot": str(prompt_snapshot),
        "results_path": str(output_path),
        "error_log_path": str(error_log_path),
        "metadata_path": str(metadata_path),
        "metric": config.metric,
        "min_delta": config.min_delta,
        "candidate": candidate_score.to_dict(),
        "baseline": baseline_score.to_dict() if baseline_score else None,
        "improved": improved,
        "promoted_to": promoted_to,
        "summary": {
            "processed": summary.processed,
            "total_results": summary.total_results,
            "null_predictions": summary.null_predictions,
        },
    }

    _write_json(run_dir / "score.json", report)
    append_jsonl(config.output_dir / "experiments.jsonl", report)
    return report
