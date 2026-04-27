#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyzing_llm_rationale.cli import build_provider, repo_root
from analyzing_llm_rationale.config import load_model_configs
from analyzing_llm_rationale.providers import (
    ProviderError,
    RetryableProviderError,
)

ROOT = repo_root()
DATASET_PATH = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
TEMPERATURE_METRICS_PATH = ROOT / "analysis" / "metrics_by_model_temperature.csv"
MODELS_CONFIG_PATH = ROOT / "configs" / "models.yaml"
OUTPUT_DIR = ROOT / "analysis" / "llm_judge_rationale_eval"
DEFAULT_TARGET_MODELS = [
    "Qwen2.5-7b-instruct",
    "Qwen3-32B",
    "GPT-OSS-120B",
]
DEFAULT_JUDGES = [
    "gemma-4-31b-it",
    "kimi-k2.5",
]
JUDGE_TEMPERATURES = {
    "gemma-4-31b-it": 0.0,
    "kimi-k2.5": 1.0,
}
DEFAULT_JUDGE_MAX_TOKENS = {
    "gemma-4-31b-it": 5000,
    "kimi-k2.5": 20000,
}
DEFAULT_JUDGE_TIMEOUTS = {
    "gemma-4-31b-it": 180.0,
    "kimi-k2.5": 600.0,
}
BATCH_SIZE = 8
MAX_WORKERS = 16
MAX_RETRIES = 5


SYSTEM_PROMPT = """You are a strict evaluator of forecasting rationales.

You will receive several forecasting questions. For each question, you will see shared context:
- question
- short description
- resolution criteria
- a short evidence digest from the provided articles

You will also see multiple model-generated rationales for the same question, one per variant.

Important rules:
1. Evaluate each rationale independently, not by direct comparison.
2. Use only the supplied context. Do not use outside knowledge.
3. Do not score forecast correctness. That is computed separately from the dataset ground truth.
4. Score each attribute on a 0.0 to 1.0 scale.
5. Return strict JSON only, with no markdown and no prose outside the JSON object.

Attribute definitions:
- plausibility: Does the rationale make coherent sense as an argument for the forecast?
- completeness: Does it cover the main evidence or decisive conditions needed for the forecast?
- source_consistency: Is it consistent with the provided question/context/evidence digest?
- non_hallucination: Does it avoid adding unsupported facts or claims not grounded in the provided context?
- informativeness: Does it provide useful, question-specific reasoning rather than generic filler?
- conciseness: Is it concise without omitting core reasoning?

Return a JSON array. Each element must have:
{
  "id": <question id>,
  "variant_scores": {
    "<variant_name>": {
      "plausibility": 0.0-1.0,
      "completeness": 0.0-1.0,
      "source_consistency": 0.0-1.0,
      "non_hallucination": 0.0-1.0,
      "informativeness": 0.0-1.0,
      "conciseness": 0.0-1.0
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two LLM judges over rationale variants for the best-temperature run "
            "of each target model. Forecast correctness is computed from dataset ground truth."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--metrics-csv", type=Path, default=TEMPERATURE_METRICS_PATH)
    parser.add_argument("--models-config", type=Path, default=MODELS_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--target-models", nargs="+", default=DEFAULT_TARGET_MODELS)
    parser.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--judge-max-tokens", type=int, default=0)
    parser.add_argument("--max-example-groups", type=int, default=0)
    parser.add_argument("--fixed-temperature", type=float, default=None)
    return parser.parse_args()


def normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value if value in {"yes", "no"} else None


def normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_f) or math.isinf(value_f) or not (0.0 <= value_f <= 1.0):
        return None
    return value_f


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def coerce_scalar_score(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        score = value.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_best_temperature_dirs(metrics_csv: Path, target_models: set[str]) -> dict[str, str]:
    best_by_model: dict[str, tuple[float, str]] = {}
    with metrics_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            if model not in target_models:
                continue
            brier = float(row["mean_brier_score"])
            temperature_dir = row["temperature_dir"]
            current = best_by_model.get(model)
            if current is None or brier < current[0]:
                best_by_model[model] = (brier, temperature_dir)
    return {model: temp_dir for model, (_, temp_dir) in best_by_model.items()}


def parse_temperature_dir(dirname: str) -> float:
    raw = dirname.removeprefix("temperature_")
    if raw in {"0", "00", "000"}:
        return 0.0
    if len(raw) == 3 and raw.isdigit():
        return int(raw) / 1000.0
    return float(raw)


def find_temperature_dir(results_root: Path, model_label: str, fixed_temperature: float) -> str:
    model_dir = results_root / model_label
    candidates = [path.name for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("temperature_")]
    for dirname in sorted(candidates):
        if abs(parse_temperature_dir(dirname) - fixed_temperature) < 1e-9:
            return dirname
    raise FileNotFoundError(
        f"No temperature directory for model {model_label} matching {fixed_temperature}"
    )


def load_dataset(dataset_path: Path) -> dict[int, dict[str, Any]]:
    rows = json.loads(dataset_path.read_text())
    dataset_by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        rid = int(row["id"])
        dataset_by_id[rid] = row
    return dataset_by_id


def build_evidence_digest(row: dict[str, Any], *, max_articles: int = 2) -> list[str]:
    articles = sorted(
        row.get("news_articles", []),
        key=lambda article: (
            coerce_scalar_score(article.get("credibility"), default=0.0),
            coerce_scalar_score(article.get("frs"), default=0.0),
        ),
        reverse=True,
    )
    digests: list[str] = []
    for article in articles[:max_articles]:
        summary = article.get("summary_llm") or article.get("summary") or ""
        credibility = coerce_scalar_score(article.get("credibility"), default=0.0)
        title = truncate(str(article.get("title") or ""), 120)
        digest = (
            f"title={title}; credibility={credibility:.2f}; "
            f"summary={truncate(str(summary), 320)}"
        )
        digests.append(digest)
    return digests


def load_variant_rows(
    results_root: Path,
    model_label: str,
    temperature_dir: str,
) -> tuple[list[str], dict[int, dict[str, dict[str, Any]]]]:
    variant_files = sorted((results_root / model_label / temperature_dir).glob("results_variant*.json"))
    variant_names = [path.stem.removeprefix("results_") for path in variant_files]
    rows_by_id: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in variant_files:
        variant = path.stem.removeprefix("results_")
        rows = json.loads(path.read_text())
        for row in rows:
            rid = int(row["id"])
            rows_by_id[rid][variant] = row
    return variant_names, rows_by_id


def build_examples(
    dataset_by_id: dict[int, dict[str, Any]],
    model_label: str,
    temperature_dir: str,
    variant_names: list[str],
    rows_by_id: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for rid, variant_rows in rows_by_id.items():
        if any(variant not in variant_rows for variant in variant_names):
            continue
        dataset_row = dataset_by_id.get(rid)
        if dataset_row is None:
            continue
        example = {
            "id": rid,
            "model": model_label,
            "temperature_dir": temperature_dir,
            "question": truncate(str(dataset_row.get("question") or ""), 260),
            "description": truncate(str(dataset_row.get("description") or ""), 420),
            "resolution_criteria": truncate(str(dataset_row.get("resolution_criteria") or ""), 420),
            "evidence_digest": build_evidence_digest(dataset_row),
            "variants": {},
        }
        answer = normalize_answer(dataset_row.get("answer"))
        for variant in variant_names:
            result_row = variant_rows[variant]
            predicted_answer = normalize_answer(result_row.get("predicted_answer"))
            confidence = normalize_confidence(result_row.get("confidence"))
            rationale = str(result_row.get("rationale") or "").strip()
            if predicted_answer is None or confidence is None or not rationale or answer is None:
                break
            example["variants"][variant] = {
                "predicted_answer": predicted_answer,
                "confidence": confidence,
                "rationale": truncate(rationale, 500),
                "forecast_correct": int(predicted_answer == answer),
            }
        else:
            examples.append(example)
    examples.sort(key=lambda item: item["id"])
    return examples


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def extract_json_array(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    raise ValueError("No JSON array found in judge response")


def variant_overall_score(score_row: dict[str, float]) -> float:
    attributes = [
        score_row["plausibility"],
        score_row["completeness"],
        score_row["source_consistency"],
        score_row["non_hallucination"],
        score_row["informativeness"],
        score_row["conciseness"],
    ]
    return sum(attributes) / len(attributes)


class JudgeRunner:
    def __init__(
        self,
        judge_name: str,
        models_config_path: Path,
        *,
        max_tokens: int,
        request_timeout_s: float,
        max_retries: int,
    ) -> None:
        self.judge_name = judge_name
        self.models_config_path = models_config_path
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self._thread_local = threading.local()

    def _build_provider(self):
        args = SimpleNamespace(
            model=self.judge_name,
            models_config=self.models_config_path,
            provider=None,
            local_model_name=None,
            router_model_name=None,
            model_label=None,
            api_base_url=None,
            api_key_env_var=None,
            api_key_file=None,
            request_timeout_s=self.request_timeout_s,
            device="cpu",
        )
        return build_provider(args)

    def provider(self):
        provider = getattr(self._thread_local, "provider", None)
        if provider is None:
            provider = self._build_provider()
            self._thread_local.provider = provider
        return provider

    def _kimi_chat_completion(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        provider = self.provider()
        payload = {
            "model": provider.model_name,
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        response = provider._session.post(
            provider.base_url,
            headers=headers,
            json=payload,
            timeout=provider.request_timeout_s,
        )
        response_text = response.text[:1000]
        if response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
            raise RetryableProviderError(
                f"status={response.status_code} body={response_text}"
            )
        if response.status_code != 200:
            raise ProviderError(
                f"status={response.status_code} body={response_text}"
            )
        try:
            message = response.json()["choices"][0]["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Malformed Kimi response: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RetryableProviderError(f"Kimi returned empty content: {response_text}")
        return content

    def score_batch(self, batch_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        user_payload = []
        for item in batch_items:
            user_payload.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "description": item["description"],
                    "resolution_criteria": item["resolution_criteria"],
                    "evidence_digest": item["evidence_digest"],
                    "variants": {
                        variant: {
                            "predicted_answer": variant_row["predicted_answer"],
                            "confidence": variant_row["confidence"],
                            "rationale": variant_row["rationale"],
                        }
                        for variant, variant_row in item["variants"].items()
                    },
                }
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        temperature = JUDGE_TEMPERATURES[self.judge_name]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                if self.judge_name == "kimi-k2.5":
                    content = self._kimi_chat_completion(messages, self.max_tokens)
                else:
                    content = self.provider().chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_tokens,
                    )
                parsed = extract_json_array(content)
                return parsed
            except (RetryableProviderError, ValueError) as exc:
                last_error = exc
                time.sleep(min(30.0, 1.5 * (2**attempt)))
            except ProviderError as exc:
                last_error = exc
                break
        if last_error is None:
            raise RuntimeError("Judge request failed without error")
        raise RuntimeError(f"Judge {self.judge_name} batch failed: {last_error}") from last_error


def load_existing_judgments(path: Path) -> set[int]:
    if not path.exists():
        return set()
    judged_ids: set[int] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            judged_ids.add(int(payload["id"]))
    return judged_ids


def run_judge_for_model(
    judge_runner: JudgeRunner,
    judge_name: str,
    model_label: str,
    temperature_dir: str,
    examples: list[dict[str, Any]],
    output_dir: Path,
    *,
    batch_size: int,
    max_workers: int,
) -> Path:
    judge_dir = output_dir / judge_name
    judge_dir.mkdir(parents=True, exist_ok=True)
    output_path = judge_dir / f"{model_label}__{temperature_dir}.jsonl"
    judged_ids = load_existing_judgments(output_path)
    pending = [item for item in examples if item["id"] not in judged_ids]
    if not pending:
        print(f"{judge_name}: {model_label} {temperature_dir} already complete")
        return output_path

    batches = batched(pending, batch_size)
    lock = threading.Lock()
    with output_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(judge_runner.score_batch, batch): batch
            for batch in batches
        }
        completed = 0
        total = len(batches)
        for future in as_completed(future_to_batch):
            parsed = future.result()
            with lock:
                for row in parsed:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(
                    f"{judge_name}: {model_label} {temperature_dir} batches {completed}/{total}",
                    flush=True,
                )
    return output_path


def parse_judge_outputs(
    dataset_by_id: dict[int, dict[str, Any]],
    output_paths: dict[tuple[str, str], Path],
    examples_by_model: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    details_rows: list[dict[str, Any]] = []
    summary_by_judge_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    aggregate_by_judge: defaultdict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    aggregate_combined: defaultdict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for (judge_name, model_label), output_path in output_paths.items():
        if not output_path.exists():
            continue
        index_by_id = {example["id"]: example for example in examples_by_model[model_label]}
        with output_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                rid = int(payload["id"])
                example = index_by_id.get(rid)
                if example is None:
                    continue
                for variant, score_row in payload["variant_scores"].items():
                    forecast_correct = int(example["variants"][variant]["forecast_correct"])
                    row = {
                        "judge": judge_name,
                        "model": model_label,
                        "temperature_dir": example["temperature_dir"],
                        "variant": variant,
                        "id": rid,
                        "forecast_correct": forecast_correct,
                        "confidence": example["variants"][variant]["confidence"],
                        "plausibility": float(score_row["plausibility"]),
                        "completeness": float(score_row["completeness"]),
                        "source_consistency": float(score_row["source_consistency"]),
                        "non_hallucination": float(score_row["non_hallucination"]),
                        "informativeness": float(score_row["informativeness"]),
                        "conciseness": float(score_row["conciseness"]),
                    }
                    row["overall_judge_score"] = variant_overall_score(row)
                    details_rows.append(row)
                    aggregate_by_judge[(judge_name, model_label, variant)].append(row)

    for (judge_name, model_label, variant), rows in aggregate_by_judge.items():
        summary_row = {
            "judge": judge_name,
            "model": model_label,
            "variant": variant,
            "n": len(rows),
            "forecast_accuracy": sum(row["forecast_correct"] for row in rows) / len(rows),
            "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
            "plausibility": sum(float(row["plausibility"]) for row in rows) / len(rows),
            "completeness": sum(float(row["completeness"]) for row in rows) / len(rows),
            "source_consistency": sum(float(row["source_consistency"]) for row in rows) / len(rows),
            "non_hallucination": sum(float(row["non_hallucination"]) for row in rows) / len(rows),
            "informativeness": sum(float(row["informativeness"]) for row in rows) / len(rows),
            "conciseness": sum(float(row["conciseness"]) for row in rows) / len(rows),
            "overall_judge_score": sum(float(row["overall_judge_score"]) for row in rows) / len(rows),
        }
        summary_by_judge_rows.append(summary_row)
        aggregate_combined[(model_label, variant)].append(summary_row)

    for (model_label, variant), rows in aggregate_combined.items():
        summary_rows.append(
            {
                "model": model_label,
                "variant": variant,
                "n_judges": len(rows),
                "forecast_accuracy": sum(float(row["forecast_accuracy"]) for row in rows) / len(rows),
                "mean_confidence": sum(float(row["mean_confidence"]) for row in rows) / len(rows),
                "plausibility": sum(float(row["plausibility"]) for row in rows) / len(rows),
                "completeness": sum(float(row["completeness"]) for row in rows) / len(rows),
                "source_consistency": sum(float(row["source_consistency"]) for row in rows) / len(rows),
                "non_hallucination": sum(float(row["non_hallucination"]) for row in rows) / len(rows),
                "informativeness": sum(float(row["informativeness"]) for row in rows) / len(rows),
                "conciseness": sum(float(row["conciseness"]) for row in rows) / len(rows),
                "overall_judge_score": sum(float(row["overall_judge_score"]) for row in rows) / len(rows),
            }
        )

    summary_by_judge_rows.sort(key=lambda row: (row["judge"], row["model"], row["variant"]))
    summary_rows.sort(key=lambda row: (row["model"], row["variant"]))
    details_rows.sort(key=lambda row: (row["judge"], row["model"], row["variant"], row["id"]))
    return details_rows, summary_by_judge_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    path: Path,
    best_temperature_dirs: dict[str, str],
    summary_rows: list[dict[str, Any]],
    target_models: list[str],
) -> None:
    lines = [
        "# Two-Judge Rationale Evaluation",
        "",
        "Best temperatures selected by minimum mean Brier score from `analysis/metrics_by_model_temperature.csv`:",
        "",
    ]
    for model in target_models:
        lines.append(f"- `{model}` -> `{best_temperature_dirs[model]}`")

    for model in target_models:
        model_rows = [row for row in summary_rows if row["model"] == model]
        model_rows.sort(key=lambda row: float(row["overall_judge_score"]), reverse=True)
        lines.extend(
            [
                "",
                f"## {model}",
                "",
                "| Variant | Judge Score | Plausibility | Completeness | Source Consistency | Non-Hallucination | Informativeness | Conciseness | Accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_rows:
            lines.append(
                f"| {row['variant']} | {float(row['overall_judge_score']):.3f} | "
                f"{float(row['plausibility']):.3f} | {float(row['completeness']):.3f} | "
                f"{float(row['source_consistency']):.3f} | {float(row['non_hallucination']):.3f} | "
                f"{float(row['informativeness']):.3f} | {float(row['conciseness']):.3f} | "
                f"{float(row['forecast_accuracy']):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_models = list(args.target_models)
    target_models_set = set(target_models)
    models_config = load_model_configs(args.models_config)
    missing_judges = [judge for judge in args.judges if judge not in models_config]
    if missing_judges:
        raise ValueError(f"Missing judge model configs: {missing_judges}")

    if args.fixed_temperature is None:
        best_temperature_dirs = load_best_temperature_dirs(args.metrics_csv, target_models_set)
    else:
        best_temperature_dirs = {
            model_label: find_temperature_dir(ROOT / "results", model_label, args.fixed_temperature)
            for model_label in target_models
        }
    dataset_by_id = load_dataset(args.dataset)

    examples_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_label in target_models:
        temperature_dir = best_temperature_dirs[model_label]
        variant_names, rows_by_id = load_variant_rows(ROOT / "results", model_label, temperature_dir)
        examples = build_examples(dataset_by_id, model_label, temperature_dir, variant_names, rows_by_id)
        if args.max_example_groups > 0:
            examples = examples[: args.max_example_groups]
        examples_by_model[model_label] = examples
        print(
            f"Prepared {len(examples)} example groups for {model_label} {temperature_dir} across {len(variant_names)} variants",
            flush=True,
        )

    output_paths: dict[tuple[str, str], Path] = {}
    for judge_name in args.judges:
        judge_max_tokens = (
            args.judge_max_tokens
            if args.judge_max_tokens > 0
            else DEFAULT_JUDGE_MAX_TOKENS.get(judge_name, 5000)
        )
        judge_request_timeout_s = DEFAULT_JUDGE_TIMEOUTS.get(judge_name, 180.0)
        judge_runner = JudgeRunner(
            judge_name,
            args.models_config,
            max_tokens=judge_max_tokens,
            request_timeout_s=judge_request_timeout_s,
            max_retries=args.max_retries,
        )
        for model_label in target_models:
            temperature_dir = best_temperature_dirs[model_label]
            output_path = run_judge_for_model(
                judge_runner,
                judge_name,
                model_label,
                temperature_dir,
                examples_by_model[model_label],
                args.output_dir,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
            )
            output_paths[(judge_name, model_label)] = output_path

    details_rows, summary_by_judge_rows, summary_rows = parse_judge_outputs(
        dataset_by_id,
        output_paths,
        examples_by_model,
    )

    write_csv(
        args.output_dir / "details.csv",
        details_rows,
        [
            "judge",
            "model",
            "temperature_dir",
            "variant",
            "id",
            "forecast_correct",
            "confidence",
            "plausibility",
            "completeness",
            "source_consistency",
            "non_hallucination",
            "informativeness",
            "conciseness",
            "overall_judge_score",
        ],
    )
    write_csv(
        args.output_dir / "summary_by_judge.csv",
        summary_by_judge_rows,
        [
            "judge",
            "model",
            "variant",
            "n",
            "forecast_accuracy",
            "mean_confidence",
            "plausibility",
            "completeness",
            "source_consistency",
            "non_hallucination",
            "informativeness",
            "conciseness",
            "overall_judge_score",
        ],
    )
    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
        [
            "model",
            "variant",
            "n_judges",
            "forecast_accuracy",
            "mean_confidence",
            "plausibility",
            "completeness",
            "source_consistency",
            "non_hallucination",
            "informativeness",
            "conciseness",
            "overall_judge_score",
        ],
    )
    write_summary_md(args.output_dir / "summary.md", best_temperature_dirs, summary_rows, target_models)
    (args.output_dir / "best_temperatures.json").write_text(
        json.dumps(best_temperature_dirs, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
