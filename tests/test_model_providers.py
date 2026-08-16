"""Unit tests for BYO model providers registry, envelope encryption, and API endpoints."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from analyzing_llm_rationale import model_providers
from analyzing_llm_rationale.model_providers import (
    CREDIBLE_PROVIDERS,
    _decrypt_provider_secret,
    _encrypt_provider_secret,
    create_chat_provider_for_config,
    delete_user_model_provider,
    get_user_provider_status_list,
    put_user_model_provider,
    read_user_model_provider,
)
from analyzing_llm_rationale.model_providers import (
    test_provider_credentials as run_test_provider_credentials,
)
from analyzing_llm_rationale.providers import (
    AnthropicProvider,
    HuggingFaceRouterProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
)
from analyzing_llm_rationale.server import _issue_session, app


class ModelProvidersTests(unittest.TestCase):
    def setUp(self) -> None:
        model_providers._memory_user_providers.clear()

    def tearDown(self) -> None:
        model_providers._memory_user_providers.clear()

    def test_credible_providers_registry(self) -> None:
        """Ensure all 16 credible model providers are registered with descriptors."""
        expected_providers = {
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "qwen",
            "kimi",
            "meta",
            "mistral",
            "xai",
            "groq",
            "together",
            "openrouter",
            "perplexity",
            "cohere",
            "huggingface",
            "custom",
        }
        self.assertEqual(set(CREDIBLE_PROVIDERS.keys()), expected_providers)

        for pid, desc in CREDIBLE_PROVIDERS.items():
            self.assertEqual(desc.id, pid)
            self.assertTrue(desc.name)
            self.assertIn(
                desc.category,
                {"frontier", "open_weights", "inference_cloud", "specialized", "custom"},
            )
            self.assertTrue(desc.default_model)
            self.assertGreaterEqual(len(desc.popular_models), 1)
            self.assertTrue(desc.default_base_url)

    def test_envelope_encryption_roundtrip(self) -> None:
        """Verify that credentials are encrypted into ciphertext and decrypted correctly."""
        user_id = "test-user-42"
        provider_id = "openai"
        secret_payload = {
            "api_key": "sk-proj-test1234567890abcdef",
            "default_model": "gpt-4o",
            "custom_base_url": "https://api.openai.com/v1",
        }

        enc = _encrypt_provider_secret(user_id, provider_id, secret_payload)
        self.assertIn("encrypted_secret", enc)
        self.assertIn("wrapped_data_key", enc)
        self.assertNotEqual(enc["encrypted_secret"], secret_payload["api_key"])
        self.assertNotIn("sk-proj", enc["encrypted_secret"])

        dec = _decrypt_provider_secret(user_id, provider_id, enc)
        self.assertEqual(dec, secret_payload)

    def test_user_model_provider_crud(self) -> None:
        """Test saving, reading, listing, status query, and deleting credentials."""
        user_id = "trader@foresea.ink"

        # 1. Initially all 16 providers are disconnected
        statuses = get_user_provider_status_list(user_id)
        self.assertEqual(len(statuses), 16)
        self.assertTrue(all(s.connected is False for s in statuses))

        # 2. Put OpenAI connection
        saved = put_user_model_provider(
            user_id=user_id,
            provider_id="openai",
            api_key="sk-openai-live-secret-key-123456",
            default_model="o3-mini",
        )
        self.assertTrue(saved.connected)
        self.assertEqual(saved.default_model, "o3-mini")
        self.assertTrue(saved.masked_key.startswith("sk-o..."))
        self.assertTrue(saved.masked_key.endswith("3456"))

        # 3. Read back
        rec = read_user_model_provider(user_id, "openai")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["default_model"], "o3-mini")
        self.assertNotIn("sk-openai", rec["encrypted_secret"])

        # 4. Status list reflects 1 connected provider
        statuses = get_user_provider_status_list(user_id)
        oa_status = next(s for s in statuses if s.provider_id == "openai")
        self.assertTrue(oa_status.connected)
        self.assertEqual(oa_status.default_model, "o3-mini")
        self.assertEqual(oa_status.masked_key, saved.masked_key)

        # 5. Delete connection
        delete_user_model_provider(user_id, "openai")
        self.assertIsNone(read_user_model_provider(user_id, "openai"))

        statuses = get_user_provider_status_list(user_id)
        oa_status = next(s for s in statuses if s.provider_id == "openai")
        self.assertFalse(oa_status.connected)

    def test_chat_provider_factory_and_resolution(self) -> None:
        """Test runtime ChatProvider instantiation across provider protocols."""
        # Anthropic
        anthropic = create_chat_provider_for_config(
            "anthropic", "sk-ant-test-key", "claude-3-7-sonnet-20250219"
        )
        self.assertIsInstance(anthropic, AnthropicProvider)
        self.assertEqual(anthropic.model_name, "claude-3-7-sonnet-20250219")

        # Hugging Face
        hf = create_chat_provider_for_config(
            "huggingface", "hf_test_token", "meta-llama/Llama-3.3-70B-Instruct"
        )
        self.assertIsInstance(hf, HuggingFaceRouterProvider)

        # OpenRouter
        openrouter = create_chat_provider_for_config("openrouter", "sk-or-test", "openrouter/auto")
        self.assertIsInstance(openrouter, OpenRouterProvider)

        # OpenAI compatible
        openai = create_chat_provider_for_config("openai", "sk-test", "gpt-4o")
        self.assertIsInstance(openai, OpenAICompatibleProvider)

        deepseek = create_chat_provider_for_config("deepseek", "sk-ds-test", "deepseek-reasoner")
        self.assertIsInstance(deepseek, OpenAICompatibleProvider)

    def test_credentials_validation_helper(self) -> None:
        """Test live credential validation helper."""
        # Unknown provider
        bad = run_test_provider_credentials("unknown-foo", "key")
        self.assertFalse(bad["ok"])
        self.assertIn("Unknown provider", bad["error"])

        # Missing required key
        missing = run_test_provider_credentials("anthropic", "")
        self.assertFalse(missing["ok"])
        self.assertIn("API key required", missing["error"])

        # Mock successful call
        with patch.object(OpenAICompatibleProvider, "chat_completion", return_value="OK"):
            res = run_test_provider_credentials("openai", "sk-test-key-12345", "gpt-4o")
            self.assertTrue(res["ok"])
            self.assertIn("Successfully connected", res["message"])
            self.assertEqual(res["sample_response"], "OK")
            self.assertGreaterEqual(res["latency_ms"], 0)

    def test_api_endpoints_flow(self) -> None:
        """Test full HTTP flow through FastAPI server."""
        client = TestClient(app)
        user_id = "user_alpha@test.com"
        token = _issue_session(user_id, user_id, "User Alpha", "")
        headers = {"Authorization": f"Bearer {token}"}

        # 1. GET /api/user/model-providers unauthenticated -> 401
        r_unauth = client.get("/api/user/model-providers")
        self.assertEqual(r_unauth.status_code, 401)

        # 2. GET /api/user/model-providers authenticated -> 200 with 16 providers
        r = client.get("/api/user/model-providers", headers=headers)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["providers"]), 16)
        self.assertTrue(all(p["connected"] is False for p in data["providers"]))

        # 3. PUT /api/user/model-providers/anthropic -> save connection
        r_save = client.put(
            "/api/user/model-providers/anthropic",
            headers=headers,
            json={
                "api_key": "sk-ant-api03-testkey-very-secret-1234567",
                "default_model": "claude-3-7-sonnet-20250219",
            },
        )
        self.assertEqual(r_save.status_code, 200)
        saved_p = r_save.json()
        self.assertTrue(saved_p["connected"])
        self.assertEqual(saved_p["provider_id"], "anthropic")
        self.assertTrue(saved_p["masked_key"].startswith("sk-a..."))

        # 4. POST /api/user/model-providers/anthropic/test -> test credentials
        with patch.object(AnthropicProvider, "chat_completion", return_value="OK"):
            r_test = client.post(
                "/api/user/model-providers/anthropic/test",
                headers=headers,
                json={},
            )
            self.assertEqual(r_test.status_code, 200)
            test_res = r_test.json()
            self.assertTrue(test_res["ok"])

        # 5. GET /chat/models returns user_models
        r_models = client.get("/chat/models", headers=headers)
        self.assertEqual(r_models.status_code, 200)
        models_data = r_models.json()
        self.assertIn("user_models", models_data)
        self.assertEqual(len(models_data["user_models"]), 1)
        self.assertEqual(models_data["user_models"][0]["provider"], "anthropic")

        # 6. DELETE /api/user/model-providers/anthropic
        r_del = client.delete("/api/user/model-providers/anthropic", headers=headers)
        self.assertEqual(r_del.status_code, 200)
        self.assertFalse(r_del.json()["connected"])


if __name__ == "__main__":
    unittest.main()
