import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.providers import (
    OpenAICompatibleProvider,
    OpenRouterProvider,
    clean_http_header_value,
)


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
