import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.providers import (
    OpenAICompatibleProvider,
    OpenRouterProvider,
    clean_http_header_value,
)


class TransportFailureClassificationTests(unittest.TestCase):
    def test_read_timeout_is_retryable_not_a_hard_failure(self):
        # Live regression: removing the agent tool loop's asyncio ceiling let
        # the HTTP client's own 120s read timeout surface instead. requests'
        # Timeout is not a ProviderError, so it escaped as a non-retryable
        # 502 and five of eight models hard-failed with no retry at all.
        import requests

        from analyzing_llm_rationale import providers

        class _TimingOutSession:
            def post(self, *_args, **_kwargs):
                raise requests.exceptions.ReadTimeout("read timed out")

        with self.assertRaises(providers.RetryableProviderError):
            providers._post(_TimingOutSession(), "https://example.invalid")

    def test_timeout_uses_the_timeout_retry_budget_not_the_general_one(self):
        # A timeout costs the full timeout window on every attempt. Budgeting
        # it as a generic retryable failure (5 attempts) instead of a timeout
        # (1) let each stuck model burn ~13 minutes and the fleet cycle was
        # cancelled having completed 1 of 8 models.
        from analyzing_llm_rationale import providers

        self.assertTrue(
            issubclass(providers.ProviderTimeoutError, providers.RetryableProviderError),
            "a timeout must still be retryable",
        )
        import requests

        class _TimingOutSession:
            def post(self, *_args, **_kwargs):
                raise requests.exceptions.ReadTimeout("read timed out")

        with self.assertRaises(providers.ProviderTimeoutError):
            providers._post(_TimingOutSession(), "https://example.invalid")

    def test_connection_error_is_retryable(self):
        import requests

        from analyzing_llm_rationale import providers

        class _DroppedSession:
            def post(self, *_args, **_kwargs):
                raise requests.exceptions.ConnectionError("connection reset")

        with self.assertRaises(providers.RetryableProviderError):
            providers._post(_DroppedSession(), "https://example.invalid")

    def test_unrelated_errors_are_not_reclassified(self):
        from analyzing_llm_rationale import providers

        class _BrokenSession:
            def post(self, *_args, **_kwargs):
                raise ValueError("programming error")

        with self.assertRaises(ValueError):
            providers._post(_BrokenSession(), "https://example.invalid")


class ProviderHeaderSanitizationTests(unittest.TestCase):
    def test_api_key_bom_is_removed_before_authorization_header(self):
        provider = OpenAICompatibleProvider(
            model_name="\ufeffopenai/gpt-oss-120b\n",
            api_key="\ufeffsk-test\n",
            base_url="\ufeffhttps://llm.scads.ai/v1\n",
        )

        headers = provider._headers()

        self.assertEqual(provider.model_name, "openai/gpt-oss-120b")
        self.assertEqual(provider.base_url, "https://llm.scads.ai/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer sk-test")
        headers["Authorization"].encode("latin-1")

    def test_embedded_invisible_header_chars_are_removed(self):
        self.assertEqual(clean_http_header_value("Bearer \ufeffsk-\u200btest\r\n"), "Bearer sk-test")

    def test_openrouter_headers_are_latin_1_encodable(self):
        provider = OpenRouterProvider(model_name="model", api_key="\ufeffsk-test")

        for value in provider._headers().values():
            value.encode("latin-1")


class ReasoningEffortPayloadTests(unittest.TestCase):
    def test_reasoning_effort_omitted_by_default(self):
        for provider in (
            OpenAICompatibleProvider(model_name="model", api_key="sk-test", base_url="https://llm.scads.ai/v1"),
            OpenRouterProvider(model_name="model", api_key="sk-test"),
        ):
            with self.subTest(provider=type(provider).__name__):
                payload = provider._payload([{"role": "user", "content": "hi"}], 0.0, 64)
                self.assertNotIn("reasoning", payload)

    def test_reasoning_effort_included_when_requested(self):
        for provider in (
            OpenAICompatibleProvider(model_name="model", api_key="sk-test", base_url="https://llm.scads.ai/v1"),
            OpenRouterProvider(model_name="model", api_key="sk-test"),
        ):
            with self.subTest(provider=type(provider).__name__):
                payload = provider._payload(
                    [{"role": "user", "content": "hi"}], 0.0, 64, reasoning_effort="high"
                )
                self.assertEqual(payload["reasoning"], {"effort": "high"})


if __name__ == "__main__":
    unittest.main()
