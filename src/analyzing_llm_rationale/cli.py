from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from analyzing_llm_rationale.config import (
    load_model_configs,
    load_variant_configs,
    temperature_to_tag,
)
from analyzing_llm_rationale.pipeline import RunConfig, process_batch
from analyzing_llm_rationale.providers import (
    HuggingFaceRouterProvider,
    LocalQwenProvider,
    OpenAICompatibleProvider,
    download_model_snapshot,
    normalize_openai_chat_completions_url,
)
from analyzing_llm_rationale.validation import (
    SchemaValidationError,
    validate_dataset_records,
    verify_result_records,
)

_GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")


def _fetch_secret_from_gcp(secret_name: str) -> Optional[str]:
    """Fetch a secret from GCP Secret Manager using the instance service account.

    Only works when running on GCP (metadata server must be reachable).
    Uses stdlib only — no google-cloud SDK required.
    """
    try:
        token_req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(token_req, timeout=2) as resp:
            token = json.loads(resp.read())["access_token"]

        secret_url = (
            f"https://secretmanager.googleapis.com/v1/projects/{_GCP_PROJECT_ID}"
            f"/secrets/{secret_name}/versions/latest:access"
        )
        secret_req = urllib.request.Request(
            secret_url, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(secret_req, timeout=5) as resp:
            payload = json.loads(resp.read())["payload"]["data"]
        return base64.b64decode(payload).decode("utf-8").strip()
    except Exception:
        return None

REMOTE_PROVIDER_CHOICES = ["local-qwen", "hf-router", "openai-compatible"]
NEWS_SOURCE_CHOICES = ["newsapi", "gdelt", "google-news", "stooq", "rss"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths() -> dict[str, Path]:
    root = repo_root()
    output_path = (
        root
        / "results"
        / "Qwen2.5-7b-instruct"
        / "temperature_00"
        / "results_variant3_reasoning_type.json"
    )
    return {
        "input_path": root / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json",
        "system_prompt_path": root / "prompts" / "system.txt",
        "user_prompt_path": root / "prompts" / "variant3_reasoning_type.txt",
        "output_path": output_path,
        "error_log_path": output_path.parent / "errors_variant3_reasoning_type.jsonl",
        "metadata_path": output_path.parent / "run_metadata_variant3_reasoning_type.json",
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(prog="analyze-llm-rationale")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-batch", help="Run the batch inference pipeline.")
    run_parser.add_argument("--provider", choices=REMOTE_PROVIDER_CHOICES, default=None)
    run_parser.add_argument("--variant", default="variant3_reasoning_type")
    run_parser.add_argument("--model", default="qwen2.5-7b-instruct")
    run_parser.add_argument("--variants-config", type=Path, default=repo_root() / "configs" / "variants.yaml")
    run_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    run_parser.add_argument("--input-path", type=Path, default=defaults["input_path"])
    run_parser.add_argument("--system-prompt-path", type=Path, default=defaults["system_prompt_path"])
    run_parser.add_argument("--user-prompt-path", type=Path, default=None)
    run_parser.add_argument("--output-path", type=Path, default=None)
    run_parser.add_argument("--error-log-path", type=Path, default=None)
    run_parser.add_argument("--run-metadata-path", type=Path, default=None)
    run_parser.add_argument(
        "--output-fields",
        default=None,
    )
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--temperature-tag", default=None)
    run_parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "2048")))
    run_parser.add_argument("--max-records", type=int, default=int(os.environ.get("MAX_RECORDS", "0")))
    run_parser.add_argument("--shard-count", type=int, default=int(os.environ.get("SHARD_COUNT", "1")))
    run_parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SHARD_INDEX", "0")))
    run_parser.add_argument("--progress-every", type=int, default=int(os.environ.get("PROGRESS_EVERY", "0")))
    run_parser.add_argument("--max-attempts", type=int, default=int(os.environ.get("RETRY_MAX", "3")))
    run_parser.add_argument(
        "--retry-base-sleep-s",
        type=float,
        default=float(os.environ.get("RETRY_BASE_SLEEP_S", "1.5")),
    )
    run_parser.add_argument("--reprocess-nulls", action="store_true")
    run_parser.add_argument("--drop-article-text", action="store_true")
    run_parser.add_argument(
        "--omit-evidence",
        action="store_true",
        help="Omit news evidence from the prompt for no-evidence baseline runs.",
    )
    run_parser.add_argument(
        "--forecast-lead-days",
        type=int,
        default=None,
        help="Forecast-time cutoff: exclude evidence published after (reference - this many days). "
        "Requires --cutoff-reference != none. Output is tagged results/<model>/<temp>/lead_<NN>d/.",
    )
    run_parser.add_argument(
        "--cutoff-reference",
        choices=["event_end", "resolve_time", "none"],
        default="none",
        help="Reference time for the forecast cutoff. 'event_end' (recommended) = earlier of "
        "named-year close and resolve_time; 'resolve_time' = formal resolution; 'none' = no cutoff "
        "(oracle-evidence upper bound, current behaviour).",
    )
    run_parser.add_argument("--model-label", default=None)
    run_parser.add_argument("--local-model-name", default=None)
    run_parser.add_argument("--router-model-name", default=None)
    run_parser.add_argument("--api-base-url", default=None)
    run_parser.add_argument("--api-key-env-var", default=None)
    run_parser.add_argument("--api-key-file", default=None)
    run_parser.add_argument("--device", default=os.environ.get("MODEL_DEVICE", "cuda"))
    run_parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "120")))

    download_parser = subparsers.add_parser("download-model", help="Download the local model.")
    download_parser.add_argument("--model", default="qwen2.5-7b-instruct")
    download_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    download_parser.add_argument("--model-name", default=None)
    download_parser.add_argument("--cache-dir", type=Path, default=None)

    smoke_parser = subparsers.add_parser("smoke-test", help="Run a simple provider smoke test.")
    smoke_parser.add_argument("--provider", choices=REMOTE_PROVIDER_CHOICES, default=None)
    smoke_parser.add_argument("--model", default="qwen2.5-7b-instruct")
    smoke_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    smoke_parser.add_argument("--local-model-name", default=None)
    smoke_parser.add_argument("--router-model-name", default=None)
    smoke_parser.add_argument("--api-base-url", default=None)
    smoke_parser.add_argument("--api-key-env-var", default=None)
    smoke_parser.add_argument("--api-key-file", default=None)
    smoke_parser.add_argument("--device", default=os.environ.get("MODEL_DEVICE", "cuda"))
    smoke_parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "120")))
    smoke_parser.add_argument("--temperature", type=float, default=0.0)
    smoke_parser.add_argument("--max-tokens", type=int, default=128)

    validate_dataset_parser = subparsers.add_parser("validate-dataset", help="Validate the input dataset schema.")
    validate_dataset_parser.add_argument("--input-path", type=Path, default=defaults["input_path"])

    verify_results_parser = subparsers.add_parser("verify-results", help="Verify result completeness and structure.")
    verify_results_parser.add_argument("--variant", default="variant3_reasoning_type")
    verify_results_parser.add_argument("--model", default="qwen2.5-7b-instruct")
    verify_results_parser.add_argument("--variants-config", type=Path, default=repo_root() / "configs" / "variants.yaml")
    verify_results_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    verify_results_parser.add_argument("--temperature", type=float, default=0.0)
    verify_results_parser.add_argument("--temperature-tag", default=None)
    verify_results_parser.add_argument("--input-path", type=Path, default=defaults["input_path"])
    verify_results_parser.add_argument("--output-path", type=Path, default=None)
    verify_results_parser.add_argument("--model-label", default=None)

    serve_parser = subparsers.add_parser("serve", help="Start FastAPI inference server.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--model", default="qwen2.5-7b-instruct")
    serve_parser.add_argument("--variant", default="variant0_neutral_baseline")
    serve_parser.add_argument("--variants-config", type=Path, default=repo_root() / "configs" / "variants.yaml")
    serve_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    serve_parser.add_argument("--temperature", type=float, default=0.0)
    serve_parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "2048")))
    serve_parser.add_argument("--provider", choices=REMOTE_PROVIDER_CHOICES, default=None)
    serve_parser.add_argument("--local-model-name", default=None)
    serve_parser.add_argument("--router-model-name", default=None)
    serve_parser.add_argument("--api-base-url", default=None)
    serve_parser.add_argument("--api-key-env-var", default=None)
    serve_parser.add_argument("--api-key-file", default=None)
    serve_parser.add_argument("--device", default=os.environ.get("MODEL_DEVICE", "cuda"))
    serve_parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "120")))
    serve_parser.add_argument("--model-label", default=None)
    serve_parser.add_argument(
        "--disable-evidence",
        action="store_true",
        help="Disable automatic news evidence retrieval for /predict requests.",
    )
    serve_parser.add_argument("--newsapi-key-env-var", default="NEWSAPI_KEY")
    serve_parser.add_argument(
        "--evidence-source",
        action="append",
        choices=NEWS_SOURCE_CHOICES,
        default=None,
        help="News source for automatic evidence. Repeat to combine sources.",
    )
    serve_parser.add_argument(
        "--disable-query-planner",
        action="store_true",
        help="Skip LangChain query planning for automatic evidence retrieval.",
    )

    fetch_rank_parser = subparsers.add_parser(
        "fetch-and-rank", help="Fetch and rank news articles for a forecasting question."
    )
    fetch_rank_parser.add_argument("--question", required=True)
    fetch_rank_parser.add_argument("--top-k", type=int, default=5)
    fetch_rank_parser.add_argument("--model", default="gpt-oss-120b")
    fetch_rank_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    fetch_rank_parser.add_argument("--api-base-url", default=None)
    fetch_rank_parser.add_argument("--api-key-env-var", default=None)
    fetch_rank_parser.add_argument("--api-key-file", default=None)
    fetch_rank_parser.add_argument("--newsapi-key-env-var", default="NEWSAPI_KEY")
    fetch_rank_parser.add_argument(
        "--source",
        action="append",
        choices=NEWS_SOURCE_CHOICES,
        default=None,
        help="News source to query. Repeat to combine sources.",
    )
    fetch_rank_parser.add_argument(
        "--disable-query-planner",
        action="store_true",
        help="Skip the LangChain query-planning step and search with the raw question.",
    )

    mcp_parser = subparsers.add_parser("mcp-server", help="Start the Foresea MCP server wrapper.")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=os.environ.get("FORESEA_MCP_TRANSPORT", "stdio"),
        help="MCP transport. Use stdio for local agent clients; streamable-http for a local HTTP MCP endpoint.",
    )
    mcp_parser.add_argument(
        "--base-url",
        default=os.environ.get("FORESEA_BASE_URL", "https://foresea.ink"),
        help="Foresea API base URL to wrap.",
    )
    mcp_parser.add_argument(
        "--api-key",
        default=os.environ.get("FORESEA_API_KEY") or os.environ.get("API_KEY"),
        help="Optional Foresea API key for deployments that require X-API-Key.",
    )
    mcp_parser.add_argument(
        "--timeout-s",
        type=float,
        default=float(os.environ.get("FORESEA_MCP_TIMEOUT_S", "120")),
        help="HTTP timeout for calls from the MCP server to Foresea.",
    )
    mcp_parser.add_argument(
        "--host",
        default=os.environ.get("FORESEA_MCP_HOST", "127.0.0.1"),
        help="Host for streamable-http or sse transports.",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FORESEA_MCP_PORT", "8000")),
        help="Port for streamable-http or sse transports.",
    )

    autoresearch_parser = subparsers.add_parser(
        "autoresearch",
        help="Run one Foresea autoresearch candidate prompt experiment.",
    )
    autoresearch_parser.add_argument("--provider", choices=REMOTE_PROVIDER_CHOICES, default=None)
    autoresearch_parser.add_argument("--model", default="gpt-oss-120b")
    autoresearch_parser.add_argument("--models-config", type=Path, default=repo_root() / "configs" / "models.yaml")
    autoresearch_parser.add_argument("--input-path", type=Path, default=defaults["input_path"])
    autoresearch_parser.add_argument("--system-prompt-path", type=Path, default=defaults["system_prompt_path"])
    autoresearch_parser.add_argument(
        "--candidate-prompt-path",
        type=Path,
        default=repo_root() / "autoresearch" / "candidate_prompt.txt",
    )
    autoresearch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "analysis" / "autoresearch",
    )
    autoresearch_parser.add_argument("--baseline-results-path", type=Path, default=None)
    autoresearch_parser.add_argument("--promote-to", type=Path, default=None)
    autoresearch_parser.add_argument(
        "--metric",
        choices=["brier_score", "accuracy", "ece"],
        default="brier_score",
    )
    autoresearch_parser.add_argument("--min-delta", type=float, default=0.0)
    autoresearch_parser.add_argument("--max-records", type=int, default=int(os.environ.get("AUTORESEARCH_MAX_RECORDS", "50")))
    autoresearch_parser.add_argument("--temperature", type=float, default=0.0)
    autoresearch_parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "1024")))
    autoresearch_parser.add_argument("--max-attempts", type=int, default=int(os.environ.get("RETRY_MAX", "2")))
    autoresearch_parser.add_argument(
        "--retry-base-sleep-s",
        type=float,
        default=float(os.environ.get("RETRY_BASE_SLEEP_S", "1.5")),
    )
    autoresearch_parser.add_argument("--full-article-text", action="store_true")
    autoresearch_parser.add_argument("--bins", type=int, default=10)
    autoresearch_parser.add_argument("--model-label", default=None)
    autoresearch_parser.add_argument("--local-model-name", default=None)
    autoresearch_parser.add_argument("--router-model-name", default=None)
    autoresearch_parser.add_argument("--api-base-url", default=None)
    autoresearch_parser.add_argument("--api-key-env-var", default=None)
    autoresearch_parser.add_argument("--api-key-file", default=None)
    autoresearch_parser.add_argument("--device", default=os.environ.get("MODEL_DEVICE", "cuda"))
    autoresearch_parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "120")))

    return parser


def resolve_model_args(args: argparse.Namespace) -> argparse.Namespace:
    models = load_model_configs(args.models_config)
    model = models[args.model]
    args._resolved_model_config = model
    if getattr(args, "provider", None) is None:
        args.provider = model.provider
    if getattr(args, "local_model_name", None) is None:
        args.local_model_name = model.local_model_name
    if getattr(args, "router_model_name", None) is None:
        args.router_model_name = model.router_model_name
    if getattr(args, "model_label", None) is None:
        args.model_label = model.result_label
    if getattr(args, "api_base_url", None) is None:
        args.api_base_url = model.api_base_url
    if getattr(args, "api_base_url", None):
        args.api_base_url = normalize_openai_chat_completions_url(args.api_base_url)
    if getattr(args, "api_key_env_var", None) is None:
        args.api_key_env_var = model.api_key_env_var
    if getattr(args, "api_key_file", None) is None:
        args.api_key_file = model.api_key_file
    return args


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.provider == "hf-router":
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or ""

    if args.api_key_env_var:
        value = os.environ.get(args.api_key_env_var)
        if value:
            return value
        # Fallback: fetch from GCP Secret Manager when running on GCP
        # (env var absent means we're likely in a Vertex AI container without baked-in creds)
        secret_value = _fetch_secret_from_gcp(args.api_key_env_var)
        if secret_value:
            return secret_value

    if args.api_key_file:
        api_key_path = Path(args.api_key_file)
        if not api_key_path.is_absolute():
            api_key_path = repo_root() / api_key_path
        try:
            return api_key_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    return ""


def build_provider(args: argparse.Namespace):
    args = resolve_model_args(args)
    model = args._resolved_model_config
    effective_request_timeout_s = args.request_timeout_s
    if model.request_timeout_cap_s is not None:
        effective_request_timeout_s = min(effective_request_timeout_s, model.request_timeout_cap_s)
    if args.provider == "local-qwen":
        return LocalQwenProvider(
            model_name=args.local_model_name,
            device=args.device,
        )
    api_key = resolve_api_key(args)
    if args.provider == "openai-compatible":
        return OpenAICompatibleProvider(
            model_name=args.router_model_name,
            api_key=api_key,
            request_timeout_s=effective_request_timeout_s,
            base_url=normalize_openai_chat_completions_url(
                args.api_base_url or "https://api.openai.com/v1/chat/completions"
            ),
            missing_api_key_message=(
                f"{args.api_key_env_var} must be set or {args.api_key_file} must exist for openai-compatible provider."
                if args.api_key_env_var or args.api_key_file
                else "API key must be set for openai-compatible provider."
            ),
        )
    return HuggingFaceRouterProvider(
        model_name=args.router_model_name,
        api_key=api_key,
        request_timeout_s=effective_request_timeout_s,
        base_url=normalize_openai_chat_completions_url(
            args.api_base_url or "https://router.huggingface.co/v1/chat/completions"
        ),
    )


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    root = repo_root()
    args = resolve_model_args(args)
    model = args._resolved_model_config
    variants = load_variant_configs(args.variants_config)
    variant = variants[args.variant]
    shard_count = max(1, getattr(args, "shard_count", 1))
    shard_index = max(0, getattr(args, "shard_index", 0))
    progress_every = max(0, getattr(args, "progress_every", 0))

    user_prompt_path = args.user_prompt_path or (root / variant.prompt_path)
    temperature_tag = args.temperature_tag or temperature_to_tag(args.temperature)
    cutoff_reference = getattr(args, "cutoff_reference", "none")
    forecast_lead_days = getattr(args, "forecast_lead_days", None)
    if cutoff_reference != "none" and forecast_lead_days is None:
        raise SystemExit("--cutoff-reference requires --forecast-lead-days")
    output_dir = root / "results" / args.model_label / temperature_tag
    if cutoff_reference != "none":
        output_dir = output_dir / f"lead_{int(forecast_lead_days)}d"
    output_path = args.output_path or (output_dir / f"results_{variant.name}.json")
    error_log_path = args.error_log_path or (output_dir / f"errors_{variant.name}.jsonl")
    run_metadata_path = getattr(args, "run_metadata_path", None) or (
        output_dir / f"run_metadata_{variant.name}.json"
    )
    if shard_count > 1 and getattr(args, "run_metadata_path", None) is None:
        run_metadata_path = output_dir / f"run_metadata_{variant.name}.shard{shard_index}.json"
    output_fields = (
        [field.strip() for field in args.output_fields.split(",") if field.strip()]
        if args.output_fields
        else list(variant.output_fields)
    )
    effective_max_tokens = args.max_tokens
    if model.max_tokens_cap is not None:
        effective_max_tokens = min(effective_max_tokens, model.max_tokens_cap)

    return RunConfig(
        input_path=args.input_path,
        output_path=output_path,
        error_log_path=error_log_path,
        system_prompt_path=args.system_prompt_path,
        user_prompt_path=user_prompt_path,
        output_fields=output_fields,
        temperature=args.temperature,
        max_tokens=effective_max_tokens,
        max_records=args.max_records,
        max_attempts=args.max_attempts,
        retry_base_sleep_s=args.retry_base_sleep_s,
        reprocess_null_only=args.reprocess_nulls,
        drop_article_text=args.drop_article_text,
        omit_evidence=getattr(args, "omit_evidence", False),
        variant_name=variant.name,
        model_key=args.model,
        model_label=args.model_label,
        provider_name=args.provider,
        model_identifier=args.local_model_name if args.provider == "local-qwen" else args.router_model_name,
        provider_base_url=args.api_base_url or "",
        temperature_tag=temperature_tag,
        run_metadata_path=run_metadata_path,
        shard_count=shard_count,
        shard_index=shard_index,
        progress_every=progress_every,
        forecast_lead_days=forecast_lead_days,
        cutoff_reference=cutoff_reference,
    )


def run_batch_command(args: argparse.Namespace) -> int:
    provider = build_provider(args)
    config = resolve_run_config(args)

    try:
        import mlflow
        _mlflow_available = True
    except ImportError:
        _mlflow_available = False

    if _mlflow_available:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
        mlflow.set_experiment("analyzing-llm-rationale")
        _run_ctx = mlflow.start_run()
        _run_ctx.__enter__()
        mlflow.log_params({
            "model_key": config.model_key,
            "variant_name": config.variant_name,
            "temperature": config.temperature,
            "max_records": config.max_records,
            "provider_name": config.provider_name,
            "max_tokens": config.max_tokens,
            "temperature_tag": config.temperature_tag,
        })

    summary = process_batch(config, provider)

    if _mlflow_available:
        mlflow.log_metrics({
            "processed": float(summary.processed),
            "null_predictions": float(summary.null_predictions),
            "total_results": float(summary.total_results),
        })
        try:
            from analyzing_llm_rationale.metrics import (
                accuracy,
                brier_score,
                ece,
                iter_examples,
                load_targets,
            )
            results = json.loads(config.output_path.read_text(encoding="utf-8"))
            targets = load_targets(config.input_path)
            examples, _ = iter_examples(results, targets)
            if examples:
                mlflow.log_metrics({
                    "accuracy": accuracy(examples),
                    "brier_score": brier_score(examples),
                    "ece": ece(examples, bins=10),
                })
        except Exception:
            pass
        if config.output_path.exists():
            mlflow.log_artifact(str(config.output_path))
        if config.run_metadata_path and config.run_metadata_path.exists():
            mlflow.log_artifact(str(config.run_metadata_path))
        _run_ctx.__exit__(None, None, None)

    print(
        f"Processed {summary.processed} records | "
        f"total={summary.total_results} | nulls={summary.null_predictions} | "
        f"output={summary.output_path}"
    )
    return 0


def download_model_command(args: argparse.Namespace) -> int:
    args = resolve_model_args(args)
    model_path = download_model_snapshot(args.model_name or args.local_model_name, cache_dir=args.cache_dir)
    print(model_path)
    return 0


def smoke_test_command(args: argparse.Namespace) -> int:
    provider = build_provider(args)
    if isinstance(provider, OpenAICompatibleProvider):
        print(f"model={provider.model_name}")
        print(f"endpoint={provider.base_url}")
    content = provider.chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one sentence."},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(content)
    return 0


def validate_dataset_command(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.input_path.read_text(encoding="utf-8"))
        validate_dataset_records(payload)
    except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
        print(f"Dataset validation failed: {exc}")
        return 1
    print(f"Dataset validation passed: {args.input_path}")
    return 0


def verify_results_command(args: argparse.Namespace) -> int:
    try:
        args = resolve_model_args(args)
        variants = load_variant_configs(args.variants_config)
        variant = variants[args.variant]
        temperature_tag = args.temperature_tag or temperature_to_tag(args.temperature)
        output_path = args.output_path or (
            repo_root() / "results" / args.model_label / temperature_tag / f"results_{variant.name}.json"
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))
        dataset = json.loads(args.input_path.read_text(encoding="utf-8"))
        summary = verify_result_records(results, variant.output_fields, dataset_records=dataset)
    except (OSError, json.JSONDecodeError, KeyError, SchemaValidationError) as exc:
        print(f"Results verification failed: {exc}")
        return 1

    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.is_clean else 1


def fetch_and_rank_command(args: argparse.Namespace) -> int:
    try:
        from analyzing_llm_rationale.news_pipeline import NewsPipeline
    except ImportError:
        print("The 'pipeline' extra is required: pip install '.[pipeline]'")
        return 1

    args = resolve_model_args(args)
    api_key = resolve_api_key(args)
    newsapi_key = os.environ.get(args.newsapi_key_env_var) if args.newsapi_key_env_var else None
    base_url = args.api_base_url or args._resolved_model_config.api_base_url or "https://llm.scads.ai/v1"
    base_url = base_url.removesuffix("/chat/completions").rstrip("/")

    pipeline = NewsPipeline(
        api_key=api_key or None,
        base_url=base_url,
        model=args.router_model_name,
        newsapi_key=newsapi_key,
        use_query_planner=not args.disable_query_planner,
        fetch_sources=args.source,
    )
    articles = pipeline.fetch_summarize_rank(args.question, top_k=args.top_k)

    print(f"\nTop {len(articles)} articles for: \"{args.question}\"\n")
    for i, a in enumerate(articles, 1):
        score = a.get("relevance_score", 0.0)
        title = a.get("title") or "(no title)"
        url = a.get("url") or ""
        summary = (a.get("summary") or "")[:200]
        print(f"  {i}. [{score:.4f}] {title}")
        if url:
            print(f"     {url}")
        if summary:
            print(f"     {summary}")
        print()
    return 0


def serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from analyzing_llm_rationale.server import _state, app
    except ImportError:
        print("The 'serve' extra is required: pip install '.[serve]'")
        return 1

    root = repo_root()
    args = resolve_model_args(args)
    _state["provider"] = build_provider(args)
    _state["variants"] = load_variant_configs(args.variants_config)
    _state["system_prompt"] = (root / "prompts" / "system.txt").read_text(encoding="utf-8").strip()
    _state["prompt_templates"] = {
        name: (root / variant.prompt_path).read_text(encoding="utf-8").strip()
        for name, variant in _state["variants"].items()
    }
    _state["temperature"] = args.temperature
    _state["max_tokens"] = args.max_tokens
    _state["model_key"] = args.model
    _state["evidence_pipeline"] = None
    if not args.disable_evidence:
        try:
            from analyzing_llm_rationale import rag
            from analyzing_llm_rationale.news_pipeline import NewsPipeline

            base_url = args.api_base_url or args._resolved_model_config.api_base_url or "https://llm.scads.ai/v1"
            base_url = base_url.removesuffix("/chat/completions").rstrip("/")
            newsapi_key = os.environ.get(args.newsapi_key_env_var) if args.newsapi_key_env_var else None
            _state["evidence_pipeline"] = NewsPipeline(
                api_key=resolve_api_key(args) or None,
                base_url=base_url,
                model=args.router_model_name,
                newsapi_key=newsapi_key,
                use_query_planner=False,
                fetch_sources=args.evidence_source or ("web", "gdelt", "google-news"),
                summarize_articles=False,
                use_embeddings=False,
                # Hybrid retrieval: dense semantic (shared mounted embedder, one
                # instance — no OOM) fused with BM25, then cross-encoder reranking
                # (tiny TinyBERT from the same mount; falls back to hybrid if absent).
                embed_fn=rag.embed,
                rerank_fn=(rag.rerank if os.environ.get("EVIDENCE_RERANK", "1") == "1" else None),
                # Drop topically-irrelevant sources (memes, unrelated PDFs) so they
                # can't ground or be cited by a forecast. Cosine scale; tunable.
                min_relevance=float(os.environ.get("EVIDENCE_MIN_RELEVANCE", "0.25")),
            )
        except Exception as exc:
            print(f"Evidence retrieval disabled: {exc}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def mcp_server_command(args: argparse.Namespace) -> int:
    try:
        from analyzing_llm_rationale.mcp_server import run_mcp_server
    except RuntimeError as exc:
        print(str(exc))
        return 1

    try:
        run_mcp_server(
            transport=args.transport,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout_s,
            host=args.host,
            port=args.port,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


def autoresearch_command(args: argparse.Namespace) -> int:
    from analyzing_llm_rationale.autoresearch import AutoresearchConfig, run_autoresearch_once

    args = resolve_model_args(args)
    provider = build_provider(args)
    report = run_autoresearch_once(
        AutoresearchConfig(
            input_path=args.input_path,
            system_prompt_path=args.system_prompt_path,
            candidate_prompt_path=args.candidate_prompt_path,
            output_dir=args.output_dir,
            baseline_results_path=args.baseline_results_path,
            promote_to=args.promote_to,
            metric=args.metric,
            max_records=args.max_records,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
            retry_base_sleep_s=args.retry_base_sleep_s,
            drop_article_text=not args.full_article_text,
            bins=args.bins,
            min_delta=args.min_delta,
            model_key=args.model,
            model_label=args.model_label,
            provider_name=args.provider,
            model_identifier=args.local_model_name if args.provider == "local-qwen" else args.router_model_name,
        ),
        provider,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    if sys.version_info < (3, 10):
        print("analyzing-llm-rationale requires Python 3.10 or newer.", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-batch":
        return run_batch_command(args)
    if args.command == "download-model":
        return download_model_command(args)
    if args.command == "smoke-test":
        return smoke_test_command(args)
    if args.command == "validate-dataset":
        return validate_dataset_command(args)
    if args.command == "verify-results":
        return verify_results_command(args)
    if args.command == "serve":
        return serve_command(args)
    if args.command == "fetch-and-rank":
        return fetch_and_rank_command(args)
    if args.command == "mcp-server":
        return mcp_server_command(args)
    if args.command == "autoresearch":
        return autoresearch_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
