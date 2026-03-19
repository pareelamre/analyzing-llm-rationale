#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzing_llm_rationale.cli import main as cli_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a configured variant through the packaged pipeline.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--provider", default=os.environ.get("RUN_PROVIDER"))
    parser.add_argument("--model", default=os.environ.get("MODEL_CONFIG", "qwen2.5-7b-instruct"))
    parser.add_argument("--model-label", default=os.environ.get("MODEL_LABEL"))
    parser.add_argument("--local-model-name", default=os.environ.get("LOCAL_MODEL_NAME"))
    parser.add_argument("--router-model-name", default=os.environ.get("ROUTER_MODEL_NAME"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.0")))
    parser.add_argument("--temperature-tag", default=os.environ.get("TEMPERATURE_TAG"))
    parser.add_argument("--max-records", type=int, default=int(os.environ.get("MAX_RECORDS", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "2048")))
    parser.add_argument("--max-attempts", type=int, default=int(os.environ.get("MAX_ATTEMPTS", os.environ.get("RETRY_MAX", "3"))))
    parser.add_argument(
        "--retry-base-sleep-s",
        type=float,
        default=float(os.environ.get("RETRY_BASE_SLEEP_S", "1.5")),
    )
    parser.add_argument("--request-timeout-s", type=float, default=float(os.environ.get("REQUEST_TIMEOUT_S", "120")))
    parser.add_argument("--device", default=os.environ.get("MODEL_DEVICE", "cuda"))
    parser.add_argument("--input-path", default=os.environ.get("INPUT_PATH"))
    parser.add_argument("--system-prompt-path", default=os.environ.get("SYSTEM_PROMPT_PATH"))
    parser.add_argument("--user-prompt-path", default=os.environ.get("PROMPT_PATH"))
    parser.add_argument("--output-path", default=os.environ.get("OUTPUT_PATH"))
    parser.add_argument("--error-log-path", default=os.environ.get("ERROR_LOG_PATH"))
    parser.add_argument("--variants-config", default=os.environ.get("VARIANTS_CONFIG"))
    parser.add_argument("--models-config", default=os.environ.get("MODELS_CONFIG"))
    parser.add_argument("--output-fields", default=os.environ.get("OUTPUT_FIELDS"))
    parser.add_argument("--reprocess-nulls", action="store_true", default=os.environ.get("REPROCESS_NULLS", "0") == "1")
    parser.add_argument("--drop-article-text", action="store_true", default=os.environ.get("DROP_ARTICLE_TEXT", "0") == "1")
    return parser


def append_optional(args_list: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    args_list.extend([flag, str(value)])


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cli_args = ["run-batch", "--variant", args.variant, "--model", args.model]
    append_optional(cli_args, "--provider", args.provider)
    append_optional(cli_args, "--model-label", args.model_label)
    append_optional(cli_args, "--local-model-name", args.local_model_name)
    append_optional(cli_args, "--router-model-name", args.router_model_name)
    append_optional(cli_args, "--temperature", args.temperature)
    append_optional(cli_args, "--temperature-tag", args.temperature_tag)
    append_optional(cli_args, "--max-records", args.max_records)
    append_optional(cli_args, "--max-tokens", args.max_tokens)
    append_optional(cli_args, "--max-attempts", args.max_attempts)
    append_optional(cli_args, "--retry-base-sleep-s", args.retry_base_sleep_s)
    append_optional(cli_args, "--request-timeout-s", args.request_timeout_s)
    append_optional(cli_args, "--device", args.device)
    append_optional(cli_args, "--input-path", args.input_path)
    append_optional(cli_args, "--system-prompt-path", args.system_prompt_path)
    append_optional(cli_args, "--user-prompt-path", args.user_prompt_path)
    append_optional(cli_args, "--output-path", args.output_path)
    append_optional(cli_args, "--error-log-path", args.error_log_path)
    append_optional(cli_args, "--variants-config", args.variants_config)
    append_optional(cli_args, "--models-config", args.models_config)
    append_optional(cli_args, "--output-fields", args.output_fields)

    if args.reprocess_nulls:
        cli_args.append("--reprocess-nulls")
    if args.drop_article_text:
        cli_args.append("--drop-article-text")

    return cli_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
