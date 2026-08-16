"""Unit tests for BYO model providers registry, envelope encryption, and API endpoints."""

from unittest.mock import patch

import pytest
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


@pytest.fixture(autouse=True)
def clean_memory_store():
    model_providers._memory_user_providers.clear()
    yield
    model_providers._memory_user_providers.clear()


def test_credible_providers_registry():
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
    assert set(CREDIBLE_PROVIDERS.keys()) == expected_providers

    for pid, desc in CREDIBLE_PROVIDERS.items():
        assert desc.id == pid
        assert desc.name
        assert desc.category in {"frontier", "open_weights", "inference_cloud", "specialized", "custom"}
        assert desc.default_model
        assert len(desc.popular_models) >= 1
        assert desc.default_base_url


def test_envelope_encryption_roundtrip():
    """Verify that credentials are encrypted into ciphertext and decrypted correctly."""
    user_id = "test-user-42"
    provider_id = "openai"
    secret_payload = {
        "api_key": "sk-proj-test1234567890abcdef",
        "default_model": "gpt-4o",
        "custom_base_url": "https://api.openai.com/v1",
    }

    enc = _encrypt_provider_secret(user_id, provider_id, secret_payload)
    assert "encrypted_secret" in enc
    assert "wrapped_data_key" in enc
    assert enc["encrypted_secret"] != secret_payload["api_key"]
    assert "sk-proj" not in enc["encrypted_secret"]

    dec = _decrypt_provider_secret(user_id, provider_id, enc)
    assert dec == secret_payload


def test_user_model_provider_crud():
    """Test saving, reading, listing, status query, and deleting credentials."""
    user_id = "trader@foresea.ink"
    
    # 1. Initially all 16 providers are disconnected
    statuses = get_user_provider_status_list(user_id)
    assert len(statuses) == 16
    assert all(s.connected is False for s in statuses)

    # 2. Put OpenAI connection
    saved = put_user_model_provider(
        user_id=user_id,
        provider_id="openai",
        api_key="sk-openai-live-secret-key-123456",
        default_model="o3-mini",
    )
    assert saved.connected is True
    assert saved.default_model == "o3-mini"
    assert saved.masked_key.startswith("sk-o...")
    assert saved.masked_key.endswith("3456")

    # 3. Read back
    rec = read_user_model_provider(user_id, "openai")
    assert rec is not None
    assert rec["default_model"] == "o3-mini"
    assert "sk-openai" not in rec["encrypted_secret"]

    # 4. Status list reflects 1 connected provider
    statuses = get_user_provider_status_list(user_id)
    oa_status = next(s for s in statuses if s.provider_id == "openai")
    assert oa_status.connected is True
    assert oa_status.default_model == "o3-mini"
    assert oa_status.masked_key == saved.masked_key

    # 5. Delete connection
    delete_user_model_provider(user_id, "openai")
    assert read_user_model_provider(user_id, "openai") is None

    statuses = get_user_provider_status_list(user_id)
    oa_status = next(s for s in statuses if s.provider_id == "openai")
    assert oa_status.connected is False


def test_chat_provider_factory_and_resolution():
    """Test runtime ChatProvider instantiation across provider protocols."""
    # Anthropic
    anthropic = create_chat_provider_for_config("anthropic", "sk-ant-test-key", "claude-3-7-sonnet-20250219")
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic.model_name == "claude-3-7-sonnet-20250219"

    # Hugging Face
    hf = create_chat_provider_for_config("huggingface", "hf_test_token", "meta-llama/Llama-3.3-70B-Instruct")
    assert isinstance(hf, HuggingFaceRouterProvider)

    # OpenRouter
    openrouter = create_chat_provider_for_config("openrouter", "sk-or-test", "openrouter/auto")
    assert isinstance(openrouter, OpenRouterProvider)

    # OpenAI compatible
    openai = create_chat_provider_for_config("openai", "sk-test", "gpt-4o")
    assert isinstance(openai, OpenAICompatibleProvider)

    deepseek = create_chat_provider_for_config("deepseek", "sk-ds-test", "deepseek-reasoner")
    assert isinstance(deepseek, OpenAICompatibleProvider)


def test_credentials_validation_helper():
    """Test live credential validation helper."""
    # Unknown provider
    bad = run_test_provider_credentials("unknown-foo", "key")
    assert bad["ok"] is False
    assert "Unknown provider" in bad["error"]

    # Missing required key
    missing = run_test_provider_credentials("anthropic", "")
    assert missing["ok"] is False
    assert "API key required" in missing["error"]

    # Mock successful call
    with patch.object(OpenAICompatibleProvider, "chat_completion", return_value="OK"):
        res = run_test_provider_credentials("openai", "sk-test-key-12345", "gpt-4o")
        assert res["ok"] is True
        assert "Successfully connected" in res["message"]
        assert res["sample_response"] == "OK"
        assert res["latency_ms"] >= 0


def test_api_endpoints_flow():
    """Test full HTTP flow through FastAPI server."""
    client = TestClient(app)
    user_id = "user_alpha@test.com"
    token = _issue_session(user_id, user_id, "User Alpha", "")
    headers = {"Authorization": f"Bearer {token}"}

    # 1. GET /api/user/model-providers unauthenticated -> 401
    r_unauth = client.get("/api/user/model-providers")
    assert r_unauth.status_code == 401

    # 2. GET /api/user/model-providers authenticated -> 200 with 16 providers
    r = client.get("/api/user/model-providers", headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert len(data["providers"]) == 16
    assert all(p["connected"] is False for p in data["providers"])

    # 3. PUT /api/user/model-providers/anthropic -> save connection
    r_save = client.put(
        "/api/user/model-providers/anthropic",
        headers=headers,
        json={
            "api_key": "sk-ant-api03-testkey-very-secret-1234567",
            "default_model": "claude-3-7-sonnet-20250219",
        },
    )
    assert r_save.status_code == 200
    saved_p = r_save.json()
    assert saved_p["connected"] is True
    assert saved_p["provider_id"] == "anthropic"
    assert saved_p["masked_key"].startswith("sk-a...")

    # 4. POST /api/user/model-providers/anthropic/test -> test credentials
    with patch.object(AnthropicProvider, "chat_completion", return_value="OK"):
        r_test = client.post(
            "/api/user/model-providers/anthropic/test",
            headers=headers,
            json={},
        )
        assert r_test.status_code == 200
        test_res = r_test.json()
        assert test_res["ok"] is True

    # 5. GET /chat/models returns user_models
    r_models = client.get("/chat/models", headers=headers)
    assert r_models.status_code == 200
    models_data = r_models.json()
    assert "user_models" in models_data
    assert len(models_data["user_models"]) == 1
    assert models_data["user_models"][0]["provider"] == "anthropic"

    # 6. DELETE /api/user/model-providers/anthropic
    r_del = client.delete("/api/user/model-providers/anthropic", headers=headers)
    assert r_del.status_code == 200
    assert r_del.json()["connected"] is False
