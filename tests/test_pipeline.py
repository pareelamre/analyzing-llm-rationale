from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.cli import build_provider, main as cli_main, resolve_run_config  # noqa: E402
from analyzing_llm_rationale.config import load_model_configs, load_variant_configs, temperature_to_tag  # noqa: E402
from analyzing_llm_rationale.pipeline import (  # noqa: E402
    RunConfig,
    build_user_prompt,
    load_json,
    parse_model_response,
    process_batch,
    recover_missing_fields,
)
from analyzing_llm_rationale.providers import (  # noqa: E402
    ContextLimitError,
    LocalQwenProvider,
    OpenAICompatibleProvider,
    ensure_prompt_fits_context,
    resolve_context_window,
)


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PipelineTests(unittest.TestCase):
    def sample_record(self):
        return {
            "id": 1,
            "question": "Will event X happen?",
            "description": "A test description",
            "resolution_criteria": "Official source says yes.",
            "categories": ["science"],
            "created_time": "2025-01-01",
            "publish_time": "2025-01-02",
            "resolve_time": "2025-06-01",
            "days_open": 30,
            "news_articles": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "authors": ["A"],
                    "publish_date": "2025-01-02",
                    "summary": "Summary",
                    "summary_llm": "LLM summary",
                    "keywords": ["k1"],
                    "frs": 0.7,
                    "credibility": "high",
                    "text": "Full article text",
                }
            ],
        }

    def test_build_user_prompt_can_drop_article_text(self):
        prompt = build_user_prompt(
            self.sample_record(),
            user_prompt_template="[question]\nReturn JSON.",
            include_article_text=False,
        )
        self.assertIn("Question: Will event X happen?", prompt)
        self.assertIn('"text": null', prompt)
        self.assertNotIn("[question]", prompt)

    def test_process_batch_skips_completed_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "input.json"
            output_path = base / "output.json"
            error_path = base / "errors.jsonl"
            system_prompt_path = base / "system.txt"
            user_prompt_path = base / "user.txt"

            records = [self.sample_record(), {**self.sample_record(), "id": 2, "question": "Will Y happen?"}]
            input_path.write_text(json.dumps(records), encoding="utf-8")
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "predicted_answer": "Yes",
                            "confidence": 0.8,
                            "rationale": "Already done.",
                            "reasoning_type": "past_trend",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            system_prompt_path.write_text("System", encoding="utf-8")
            user_prompt_path.write_text("[question]\nReturn JSON.", encoding="utf-8")

            provider = FakeProvider(
                [
                    json.dumps(
                        {
                            "predicted_answer": "No",
                            "confidence": 0.4,
                            "rationale": "Fresh result.",
                            "reasoning_type": "speculation",
                        }
                    )
                ]
            )

            summary = process_batch(
                RunConfig(
                    input_path=input_path,
                    output_path=output_path,
                    error_log_path=error_path,
                    system_prompt_path=system_prompt_path,
                    user_prompt_path=user_prompt_path,
                    output_fields=("predicted_answer", "confidence", "rationale", "reasoning_type"),
                    max_attempts=1,
                ),
                provider,
            )

            self.assertEqual(summary.processed, 1)
            self.assertEqual(len(provider.calls), 1)
            results = load_json(output_path)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["id"], 1)
            self.assertEqual(results[1]["id"], 2)

    def test_process_batch_retries_after_context_trim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "input.json"
            output_path = base / "output.json"
            error_path = base / "errors.jsonl"
            system_prompt_path = base / "system.txt"
            user_prompt_path = base / "user.txt"

            input_path.write_text(json.dumps([self.sample_record()]), encoding="utf-8")
            system_prompt_path.write_text("System", encoding="utf-8")
            user_prompt_path.write_text("[question]\nReturn JSON.", encoding="utf-8")

            provider = FakeProvider(
                [
                    ContextLimitError("too long"),
                    json.dumps(
                        {
                            "predicted_answer": "Yes",
                            "confidence": 0.7,
                            "rationale": "Trimmed prompt succeeded.",
                            "reasoning_type": "stated_plan",
                        }
                    ),
                ]
            )

            process_batch(
                RunConfig(
                    input_path=input_path,
                    output_path=output_path,
                    error_log_path=error_path,
                    system_prompt_path=system_prompt_path,
                    user_prompt_path=user_prompt_path,
                    output_fields=("predicted_answer", "confidence", "rationale", "reasoning_type"),
                    max_attempts=2,
                ),
                provider,
            )

            self.assertEqual(len(provider.calls), 2)
            self.assertIn('"text": "Full article text"', provider.calls[0][1]["content"])
            self.assertIn('"text": null', provider.calls[1][1]["content"])
            results = load_json(output_path)
            self.assertEqual(results[0]["predicted_answer"], "Yes")
            error_log = error_path.read_text(encoding="utf-8")
            self.assertIn("context_trim", error_log)

    def test_reprocess_null_only_updates_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "input.json"
            output_path = base / "output.json"
            error_path = base / "errors.jsonl"
            system_prompt_path = base / "system.txt"
            user_prompt_path = base / "user.txt"

            input_path.write_text(json.dumps([self.sample_record()]), encoding="utf-8")
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "predicted_answer": None,
                            "confidence": None,
                            "rationale": None,
                            "reasoning_type": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            system_prompt_path.write_text("System", encoding="utf-8")
            user_prompt_path.write_text("[question]\nReturn JSON.", encoding="utf-8")

            provider = FakeProvider(
                [
                    json.dumps(
                        {
                            "predicted_answer": "No",
                            "confidence": 0.2,
                            "rationale": "Updated from null.",
                            "reasoning_type": "expert_forecast",
                        }
                    )
                ]
            )

            process_batch(
                RunConfig(
                    input_path=input_path,
                    output_path=output_path,
                    error_log_path=error_path,
                    system_prompt_path=system_prompt_path,
                    user_prompt_path=user_prompt_path,
                    output_fields=("predicted_answer", "confidence", "rationale", "reasoning_type"),
                    max_attempts=1,
                    reprocess_null_only=True,
                    drop_article_text=True,
                ),
                provider,
            )

            self.assertIn('"text": null', provider.calls[0][1]["content"])
            results = load_json(output_path)
            self.assertEqual(results[0]["predicted_answer"], "No")

    def test_parse_model_response_recovers_nested_json_from_rationale(self):
        content = json.dumps(
            {
                "predicted_answer": None,
                "confidence": None,
                "rationale": json.dumps(
                    {
                        "predicted_answer": "No",
                        "confidence": 0.85,
                        "rationale": "Recovered from nested JSON.",
                    }
                ),
            }
        )

        parsed = parse_model_response(content, ("predicted_answer", "confidence", "rationale"))

        self.assertEqual(parsed["predicted_answer"], "No")
        self.assertEqual(parsed["confidence"], 0.85)
        self.assertEqual(parsed["rationale"], "Recovered from nested JSON.")

    def test_recover_missing_fields_keeps_existing_non_null_values(self):
        parsed = recover_missing_fields(
            {
                "predicted_answer": "Yes",
                "confidence": None,
                "rationale": json.dumps(
                    {
                        "predicted_answer": "No",
                        "confidence": 0.42,
                        "rationale": "Nested rationale.",
                    }
                ),
            },
            ("predicted_answer", "confidence", "rationale"),
        )

        self.assertEqual(parsed["predicted_answer"], "Yes")
        self.assertEqual(parsed["confidence"], 0.42)
        self.assertEqual(parsed["rationale"], "Nested rationale.")

    def test_recover_missing_fields_handles_jsonish_rationale_with_literal_newlines(self):
        parsed = recover_missing_fields(
            {
                "predicted_answer": None,
                "confidence": None,
                "rationale": (
                    '{\n'
                    '  "predicted_answer": "No",\n'
                    '  "confidence": 0.85,\n'
                    '  "rationale": "Recovered explanation.",\n'
                    '  "text": "line one\nline two"\n'
                    '}'
                ),
            },
            ("predicted_answer", "confidence", "rationale"),
        )

        self.assertEqual(parsed["predicted_answer"], "No")
        self.assertEqual(parsed["confidence"], 0.85)
        self.assertEqual(parsed["rationale"], "Recovered explanation.")

    def test_config_loaders_expose_variant_and_model_metadata(self):
        repo_root = Path(__file__).resolve().parents[1]
        variants = load_variant_configs(repo_root / "configs" / "variants.yaml")
        models = load_model_configs(repo_root / "configs" / "models.yaml")

        self.assertIn("variant6_step_by_step_reasoning", variants)
        self.assertEqual(variants["variant6_step_by_step_reasoning"].output_fields[-1], "steps")
        self.assertIn("qwen2.5-7b-instruct", models)
        self.assertEqual(models["qwen2.5-7b-instruct"].local_model_name, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIn("llama-3.3-70b-instruct", models)
        self.assertEqual(models["llama-3.3-70b-instruct"].provider, "openai-compatible")
        self.assertEqual(
            models["llama-3.3-70b-instruct"].router_model_name,
            "meta-llama/Llama-3.3-70B-Instruct",
        )
        self.assertEqual(
            models["llama-3.3-70b-instruct"].api_base_url,
            "https://llm.scads.ai/v1/chat/completions",
        )
        self.assertEqual(models["llama-3.3-70b-instruct"].api_key_file, "SCADS_AI_API_KEY.txt")
        self.assertEqual(temperature_to_tag(0.7), "temperature_07")

    def test_resolve_run_config_builds_output_path_from_variant_model_and_temperature(self):
        repo_root = Path(__file__).resolve().parents[1]
        args = Namespace(
            provider="local-qwen",
            variant="variant5_key_conditions",
            model="qwen2.5-7b-instruct",
            variants_config=repo_root / "configs" / "variants.yaml",
            models_config=repo_root / "configs" / "models.yaml",
            input_path=repo_root / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json",
            system_prompt_path=repo_root / "prompts" / "system.txt",
            user_prompt_path=None,
            output_path=None,
            error_log_path=None,
            output_fields=None,
            temperature=0.7,
            temperature_tag=None,
            max_tokens=1024,
            max_records=10,
            max_attempts=2,
            retry_base_sleep_s=1.5,
            reprocess_nulls=False,
            drop_article_text=False,
            model_label=None,
            local_model_name=None,
            router_model_name=None,
            device="cuda",
            request_timeout_s=120.0,
        )

        config = resolve_run_config(args)

        self.assertTrue(str(config.user_prompt_path).endswith("prompts/variant5_key_conditions.txt"))
        self.assertTrue(
            str(config.output_path).endswith(
                "results/Qwen2.5-7b-instruct/temperature_07/results_variant5_key_conditions.json"
            )
        )
        self.assertEqual(config.output_fields[-1], "key_conditions")

    def test_context_window_resolution_prefers_config_then_model_then_tokenizer(self):
        self.assertEqual(resolve_context_window(16000, 32000, 64000), 16000)
        self.assertEqual(resolve_context_window(None, 32000, 64000), 32000)
        self.assertEqual(resolve_context_window(None, None, 32768), 32768)
        self.assertIsNone(resolve_context_window(None, None, 10**12))

    def test_context_window_preflight_raises_before_generation(self):
        with self.assertRaises(ContextLimitError):
            ensure_prompt_fits_context(input_tokens=31000, max_tokens=2048, context_window=32000)
        ensure_prompt_fits_context(input_tokens=1000, max_tokens=512, context_window=32000)

    def test_local_qwen_greedy_run_neutralizes_sampling_defaults(self):
        class FakeTensor:
            def __init__(self, values):
                self.shape = (1, len(values))
                self.values = values

        class FakeBatch(dict):
            def __init__(self, values):
                super().__init__()
                self.input_ids = FakeTensor(values)
                self["input_ids"] = self.input_ids

            def to(self, _device):
                return self

        class FakeTokenizer:
            pad_token_id = 7
            eos_token_id = 9
            model_max_length = 4096

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

            def __call__(self, text, return_tensors):
                return FakeBatch([1, 2, 3])

            def decode(self, token_ids, skip_special_tokens):
                return "decoded"

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(max_position_embeddings=4096)
                self.generation_config = SimpleNamespace(
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    typical_p=0.95,
                )
                self.last_generate_kwargs = None

            def generate(self, **kwargs):
                self.last_generate_kwargs = kwargs
                return [[1, 2, 3, 4, 5]]

        provider = LocalQwenProvider()
        provider._model = FakeModel()
        provider._tokenizer = FakeTokenizer()
        provider._resolved_device = "cpu"

        result = provider.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=32,
        )

        self.assertEqual(result, "decoded")
        generation_config = provider._model.last_generate_kwargs["generation_config"]
        self.assertFalse(generation_config.do_sample)
        self.assertEqual(generation_config.temperature, 1.0)
        self.assertEqual(generation_config.top_p, 1.0)
        self.assertEqual(generation_config.top_k, 50)
        self.assertEqual(generation_config.typical_p, 1.0)

    def test_cli_run_batch_writes_results_metadata_and_verifies_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            input_path = base / "input.json"
            system_prompt_path = base / "system.txt"
            prompt_path = base / "prompt.txt"
            output_path = base / "results_variant_test.json"
            error_log_path = base / "errors_variant_test.jsonl"
            metadata_path = base / "run_metadata_variant_test.json"
            variants_config_path = base / "variants.yaml"
            models_config_path = base / "models.yaml"

            input_path.write_text(json.dumps([self.sample_record()]), encoding="utf-8")
            system_prompt_path.write_text("System prompt", encoding="utf-8")
            prompt_path.write_text("[question]\nReturn JSON.", encoding="utf-8")
            variants_config_path.write_text(
                json.dumps(
                    {
                        "variants": {
                            "variant_test": {
                                "prompt_path": str(prompt_path),
                                "output_fields": ["predicted_answer", "confidence", "rationale"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            models_config_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "test-model": {
                                "result_label": "TestModel",
                                "provider": "local-qwen",
                                "local_model_name": "example/local",
                                "router_model_name": "example/router",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            provider = FakeProvider(
                [
                    json.dumps(
                        {
                            "predicted_answer": "Yes",
                            "confidence": 0.61,
                            "rationale": "Mocked integration response.",
                        }
                    )
                ]
            )

            with patch("analyzing_llm_rationale.cli.build_provider", return_value=provider):
                exit_code = cli_main(
                    [
                        "run-batch",
                        "--variant",
                        "variant_test",
                        "--model",
                        "test-model",
                        "--variants-config",
                        str(variants_config_path),
                        "--models-config",
                        str(models_config_path),
                        "--input-path",
                        str(input_path),
                        "--system-prompt-path",
                        str(system_prompt_path),
                        "--output-path",
                        str(output_path),
                        "--error-log-path",
                        str(error_log_path),
                        "--run-metadata-path",
                        str(metadata_path),
                        "--temperature",
                        "0.3",
                    ]
                )

            self.assertEqual(exit_code, 0)
            results = load_json(output_path)
            self.assertEqual(results[0]["predicted_answer"], "Yes")
            metadata = load_json(metadata_path)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["variant"], "variant_test")
            self.assertEqual(metadata["summary"]["processed"], 1)

            verify_exit_code = cli_main(
                [
                    "verify-results",
                    "--variant",
                    "variant_test",
                    "--model",
                    "test-model",
                    "--variants-config",
                    str(variants_config_path),
                    "--models-config",
                    str(models_config_path),
                    "--input-path",
                    str(input_path),
                    "--output-path",
                    str(output_path),
                    "--temperature",
                    "0.3",
                ]
            )
            self.assertEqual(verify_exit_code, 0)

    def test_build_provider_reads_openai_compatible_key_from_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            api_key_file = base / "scads-key.txt"
            api_key_file.write_text("test-key\n", encoding="utf-8")
            models_config_path = base / "models.yaml"
            models_config_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "test-model": {
                                "result_label": "TestModel",
                                "provider": "openai-compatible",
                                "local_model_name": "example/local",
                                "router_model_name": "example/remote",
                                "api_base_url": "https://llm.scads.ai/v1/chat/completions",
                                "api_key_file": str(api_key_file),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                provider=None,
                model="test-model",
                models_config=models_config_path,
                local_model_name=None,
                router_model_name=None,
                model_label=None,
                api_base_url=None,
                api_key_env_var=None,
                api_key_file=None,
                device="cuda",
                request_timeout_s=30.0,
            )

            provider = build_provider(args)

            self.assertIsInstance(provider, OpenAICompatibleProvider)
            self.assertEqual(provider.api_key, "test-key")
            self.assertEqual(provider.model_name, "example/remote")
            self.assertEqual(provider.base_url, "https://llm.scads.ai/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
