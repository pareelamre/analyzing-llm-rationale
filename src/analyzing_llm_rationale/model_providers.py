"""BYO Model Provider connections, secure envelope encryption, testing, and runtime dispatch."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from analyzing_llm_rationale.providers import (
    AnthropicProvider,
    ChatProvider,
    HuggingFaceRouterProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    clean_http_header_value,
    clean_provider_string,
)

_MODEL_PROVIDER_ENVELOPE_VERSION = 1
_KMS_LOCATION = "global"
_KMS_KEY_RING = "foresea-app-ring"
_KMS_KEY_NAME = "foresea-app-key"
_MODEL_PROVIDER_KIND = "UserModelProvider"

# ── Credible Model Providers Registry ─────────────────────────────────────────

@dataclass(frozen=True)
class ProviderDescriptor:
    id: str
    name: str
    category: str  # "frontier", "open_weights", "inference_cloud", "specialized", "custom"
    description: str
    default_model: str
    popular_models: List[str]
    default_base_url: str
    key_prefix: str
    docs_url: str
    wire_protocol: str = "openai_compatible"  # "openai_compatible", "anthropic_messages"
    requires_key: bool = True
    supports_custom_base_url: bool = True


CREDIBLE_PROVIDERS: Dict[str, ProviderDescriptor] = {
    "openai": ProviderDescriptor(
        id="openai",
        name="OpenAI",
        category="frontier",
        description="Industry-standard GPT-4o, o1, o3-mini, and GPT-4.5 models.",
        default_model="gpt-4o",
        popular_models=["gpt-4o", "gpt-4o-mini", "o1", "o3-mini", "gpt-4.5-preview"],
        default_base_url="https://api.openai.com/v1",
        key_prefix="sk-",
        docs_url="https://platform.openai.com/api-keys",
        wire_protocol="openai_compatible",
    ),
    "anthropic": ProviderDescriptor(
        id="anthropic",
        name="Anthropic",
        category="frontier",
        description="Claude 3.7 Sonnet (hybrid reasoning), Claude 3.5 Sonnet, and Claude 3.5 Haiku.",
        default_model="claude-3-7-sonnet-20250219",
        popular_models=["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        default_base_url="https://api.anthropic.com/v1/messages",
        key_prefix="sk-ant-",
        docs_url="https://console.anthropic.com/settings/keys",
        wire_protocol="anthropic_messages",
    ),
    "google": ProviderDescriptor(
        id="google",
        name="Google Gemini",
        category="frontier",
        description="Gemini 2.0 Flash, Gemini 2.0 Pro Exp, and Gemini 1.5 Pro via Google AI Studio.",
        default_model="gemini-2.0-flash",
        popular_models=["gemini-2.0-flash", "gemini-2.0-pro-exp-02-05", "gemini-1.5-pro", "gemini-1.5-flash"],
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_prefix="AIza",
        docs_url="https://aistudio.google.com/app/apikey",
        wire_protocol="openai_compatible",
    ),
    "deepseek": ProviderDescriptor(
        id="deepseek",
        name="DeepSeek",
        category="open_weights",
        description="DeepSeek-V3 and DeepSeek-R1 deep reasoning models via DeepSeek Open Platform.",
        default_model="deepseek-chat",
        popular_models=["deepseek-chat", "deepseek-reasoner"],
        default_base_url="https://api.deepseek.com/v1",
        key_prefix="sk-",
        docs_url="https://platform.deepseek.com/api_keys",
        wire_protocol="openai_compatible",
    ),
    "qwen": ProviderDescriptor(
        id="qwen",
        name="Qwen (Alibaba DashScope)",
        category="open_weights",
        description="Qwen 2.5 72B, Qwen Max, Qwen Plus, and QwQ reasoning via Alibaba Cloud DashScope.",
        default_model="qwen-max",
        popular_models=["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct", "qwq-32b-preview"],
        default_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_prefix="sk-",
        docs_url="https://dashscope.console.aliyun.com/apiKey",
        wire_protocol="openai_compatible",
    ),
    "kimi": ProviderDescriptor(
        id="kimi",
        name="Kimi (Moonshot AI)",
        category="frontier",
        description="Moonshot AI long-context and Kimi reasoning models.",
        default_model="moonshot-v1-8k",
        popular_models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-latest"],
        default_base_url="https://api.moonshot.cn/v1",
        key_prefix="sk-",
        docs_url="https://platform.moonshot.cn/console/api-keys",
        wire_protocol="openai_compatible",
    ),
    "meta": ProviderDescriptor(
        id="meta",
        name="Meta (Llama)",
        category="open_weights",
        description="Llama 3.3 70B, Llama 3.1 405B, and Llama 3.1 8B via Groq/OpenRouter/Together.",
        default_model="llama-3.3-70b-versatile",
        popular_models=["llama-3.3-70b-versatile", "llama-3.1-405b", "llama-3.1-8b-instant"],
        default_base_url="https://api.groq.com/openai/v1",
        key_prefix="gsk_",
        docs_url="https://console.groq.com/keys",
        wire_protocol="openai_compatible",
    ),
    "mistral": ProviderDescriptor(
        id="mistral",
        name="Mistral AI",
        category="frontier",
        description="Mistral Large, Pixtral Large, Codestral, and Mistral Small via La Plateforme.",
        default_model="mistral-large-latest",
        popular_models=["mistral-large-latest", "pixtral-large-latest", "codestral-latest", "mistral-small-latest"],
        default_base_url="https://api.mistral.ai/v1",
        key_prefix="",
        docs_url="https://console.mistral.ai/api-keys",
        wire_protocol="openai_compatible",
    ),
    "xai": ProviderDescriptor(
        id="xai",
        name="xAI (Grok)",
        category="frontier",
        description="Grok 2 and Grok Vision frontier models via xAI platform.",
        default_model="grok-2-latest",
        popular_models=["grok-2-latest", "grok-2-vision-1212", "grok-beta"],
        default_base_url="https://api.x.ai/v1",
        key_prefix="xai-",
        docs_url="https://console.x.ai",
        wire_protocol="openai_compatible",
    ),
    "groq": ProviderDescriptor(
        id="groq",
        name="Groq",
        category="inference_cloud",
        description="Ultra-low latency LPU inference for Llama 3.3, DeepSeek R1 Distill, and Gemma 2.",
        default_model="llama-3.3-70b-versatile",
        popular_models=["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"],
        default_base_url="https://api.groq.com/openai/v1",
        key_prefix="gsk_",
        docs_url="https://console.groq.com/keys",
        wire_protocol="openai_compatible",
    ),
    "together": ProviderDescriptor(
        id="together",
        name="Together AI",
        category="inference_cloud",
        description="Fast open-source inference for Llama 3.3, DeepSeek V3, and Qwen 2.5.",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        popular_models=["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        default_base_url="https://api.together.xyz/v1",
        key_prefix="",
        docs_url="https://api.together.ai/settings/api-keys",
        wire_protocol="openai_compatible",
    ),
    "openrouter": ProviderDescriptor(
        id="openrouter",
        name="OpenRouter",
        category="inference_cloud",
        description="Unified API routing across 200+ models from every major provider.",
        default_model="openrouter/auto",
        popular_models=["openrouter/auto", "anthropic/claude-3.7-sonnet", "deepseek/deepseek-r1", "openai/gpt-4o"],
        default_base_url="https://openrouter.ai/api/v1",
        key_prefix="sk-or-",
        docs_url="https://openrouter.ai/keys",
        wire_protocol="openai_compatible",
    ),
    "perplexity": ProviderDescriptor(
        id="perplexity",
        name="Perplexity AI",
        category="specialized",
        description="Online search-grounded reasoning models (Sonar Reasoning Pro, Sonar Pro).",
        default_model="sonar-reasoning-pro",
        popular_models=["sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar"],
        default_base_url="https://api.perplexity.ai",
        key_prefix="pplx-",
        docs_url="https://www.perplexity.ai/settings/api",
        wire_protocol="openai_compatible",
    ),
    "cohere": ProviderDescriptor(
        id="cohere",
        name="Cohere",
        category="specialized",
        description="Command R+ and Command R enterprise reasoning and tool-use models.",
        default_model="command-r-plus-08-2024",
        popular_models=["command-r-plus-08-2024", "command-r-08-2024"],
        default_base_url="https://api.cohere.com/v2",
        key_prefix="",
        docs_url="https://dashboard.cohere.com/api-keys",
        wire_protocol="openai_compatible",
    ),
    "huggingface": ProviderDescriptor(
        id="huggingface",
        name="Hugging Face",
        category="inference_cloud",
        description="Hugging Face Serverless Inference and Dedicated Endpoints.",
        default_model="meta-llama/Llama-3.3-70B-Instruct",
        popular_models=["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1"],
        default_base_url="https://router.huggingface.co/v1",
        key_prefix="hf_",
        docs_url="https://huggingface.co/settings/tokens",
        wire_protocol="openai_compatible",
    ),
    "custom": ProviderDescriptor(
        id="custom",
        name="Custom / Local / Self-Hosted",
        category="custom",
        description="Connect any custom vLLM, Ollama, TGI, SGLang, or private OpenAI-compatible endpoint.",
        default_model="custom-model",
        popular_models=["custom-model"],
        default_base_url="http://localhost:11434/v1",
        key_prefix="",
        docs_url="",
        wire_protocol="openai_compatible",
        requires_key=False,
    ),
}

# ── Dataclasses for Client & API Transfer ─────────────────────────────────────

@dataclass
class ModelProviderStatus:
    provider_id: str
    name: str
    category: str
    description: str
    connected: bool
    default_model: str
    popular_models: List[str]
    default_base_url: str
    custom_base_url: Optional[str] = None
    docs_url: str = ""
    updated_at: Optional[str] = None
    key_prefix: str = ""
    masked_key: Optional[str] = None


@dataclass
class ModelProvidersResponse:
    providers: List[ModelProviderStatus]
    encryption_configured: bool = True


# ── Encryption & Storage Helpers ──────────────────────────────────────────────

_kms_client = None

def _get_kms_client() -> Any:
    global _kms_client
    if not (os.environ.get("K_SERVICE") or os.environ.get("ENABLE_CLOUD_KMS") == "1"):
        return None
    if _kms_client is None:
        try:
            from google.cloud import kms_v1
            _kms_client = kms_v1.KeyManagementServiceClient()
        except Exception:
            _kms_client = None
    return _kms_client


def _kms_key_name() -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "foresea-app"
    return f"projects/{project_id}/locations/{_KMS_LOCATION}/keyRings/{_KMS_KEY_RING}/cryptoKeys/{_KMS_KEY_NAME}"


def _kms_aad(user_id: str, provider_id: str) -> bytes:
    return f"foresea:user_model_provider:{user_id}:{provider_id}".encode("utf-8")


def _fallback_fernet_key() -> bytes:
    seed = os.environ.get("SESSION_SECRET_KEY") or "foresea-model-providers-local-dev-fallback-key-32b"
    import hashlib
    key_32 = hashlib.sha256(seed.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key_32)


def _encrypt_provider_secret(user_id: str, provider_id: str, secret_payload: Dict[str, Any]) -> Dict[str, Any]:
    from cryptography.fernet import Fernet

    data_key = Fernet.generate_key()
    plaintext = json.dumps(
        {
            "version": _MODEL_PROVIDER_ENVELOPE_VERSION,
            "user_id": user_id,
            "provider_id": provider_id,
            "payload": secret_payload,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    ciphertext = Fernet(data_key).encrypt(plaintext).decode("utf-8")

    kms = _get_kms_client()
    kms_key_name = _kms_key_name()
    wrapped_key_str = ""
    if kms is not None:
        try:
            wrapped = kms.encrypt(
                request={
                    "name": kms_key_name,
                    "plaintext": data_key,
                    "additional_authenticated_data": _kms_aad(user_id, provider_id),
                }
            )
            wrapped_key_str = base64.urlsafe_b64encode(wrapped.ciphertext).decode("ascii")
        except Exception:
            kms = None  # Fallback to local envelope

    if not wrapped_key_str:
        wrapped_key_str = Fernet(_fallback_fernet_key()).encrypt(data_key).decode("ascii")

    return {
        "encrypted_secret": ciphertext,
        "wrapped_data_key": wrapped_key_str,
        "kms_key_name": kms_key_name if kms is not None else "local-fallback",
        "credential_version": _MODEL_PROVIDER_ENVELOPE_VERSION,
    }


def _decrypt_provider_secret(user_id: str, provider_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    from cryptography.fernet import Fernet

    ciphertext = record.get("encrypted_secret")
    wrapped_data_key_str = record.get("wrapped_data_key")
    if not ciphertext or not wrapped_data_key_str:
        raise ValueError("Malformed encrypted provider record")

    kms_key_name = record.get("kms_key_name", "")
    kms = _get_kms_client() if kms_key_name != "local-fallback" else None

    data_key = None
    if kms is not None:
        try:
            wrapped_bytes = base64.urlsafe_b64decode(wrapped_data_key_str.encode("ascii"))
            data_key = kms.decrypt(
                request={
                    "name": kms_key_name,
                    "ciphertext": wrapped_bytes,
                    "additional_authenticated_data": _kms_aad(user_id, provider_id),
                }
            ).plaintext
        except Exception:
            data_key = None

    if data_key is None:
        data_key = Fernet(_fallback_fernet_key()).decrypt(wrapped_data_key_str.encode("ascii"))

    plaintext = Fernet(data_key).decrypt(ciphertext.encode("utf-8"))
    envelope = json.loads(plaintext.decode("utf-8"))
    return envelope.get("payload", {})


# ── Persistence (Datastore / Memory) ──────────────────────────────────────────

_memory_user_providers: Dict[str, Dict[str, Dict[str, Any]]] = {}

def _get_datastore_client() -> Any:
    if not (os.environ.get("K_SERVICE") or os.environ.get("DATASTORE_EMULATOR_HOST") or os.environ.get("ENABLE_DATASTORE") == "1"):
        return None
    try:
        from google.cloud import datastore
        return datastore.Client()
    except Exception:
        return None


def _model_provider_key(client: Any, user_id: str, provider_id: str) -> Any:
    return client.key("User", user_id, _MODEL_PROVIDER_KIND, provider_id)


def read_user_model_provider(user_id: str, provider_id: str) -> Optional[Dict[str, Any]]:
    client = _get_datastore_client()
    if client is None:
        rec = _memory_user_providers.get(user_id, {}).get(provider_id)
        return dict(rec) if rec else None
    key = _model_provider_key(client, user_id, provider_id)
    entity = client.get(key)
    return dict(entity) if entity else None


def list_user_model_providers(user_id: str) -> List[Dict[str, Any]]:
    client = _get_datastore_client()
    if client is None:
        return list(_memory_user_providers.get(user_id, {}).values())
    query = client.query(kind=_MODEL_PROVIDER_KIND, ancestor=client.key("User", user_id))
    return [dict(e) for e in query.fetch()]


def put_user_model_provider(
    user_id: str,
    provider_id: str,
    api_key: str,
    default_model: Optional[str] = None,
    custom_base_url: Optional[str] = None,
) -> ModelProviderStatus:
    provider = CREDIBLE_PROVIDERS.get(provider_id)
    if not provider:
        raise ValueError(f"Unknown provider: {provider_id}")

    api_key = clean_http_header_value(api_key)
    if provider.requires_key and not api_key:
        raise ValueError(f"API key is required for {provider.name}")

    now = datetime.now(timezone.utc).isoformat()
    selected_model = clean_provider_string(default_model) or provider.default_model
    selected_url = clean_provider_string(custom_base_url) or provider.default_base_url

    secret_payload = {
        "api_key": api_key,
        "default_model": selected_model,
        "custom_base_url": selected_url,
    }
    enc = _encrypt_provider_secret(user_id, provider_id, secret_payload)
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) >= 12 else ("****" if api_key else "")

    record = {
        "provider_id": provider_id,
        "default_model": selected_model,
        "custom_base_url": selected_url,
        "masked_key": masked_key,
        "encrypted_secret": enc["encrypted_secret"],
        "wrapped_data_key": enc["wrapped_data_key"],
        "kms_key_name": enc["kms_key_name"],
        "credential_version": enc["credential_version"],
        "updated_at": now,
    }

    client = _get_datastore_client()
    if client is None:
        _memory_user_providers.setdefault(user_id, {})[provider_id] = record
    else:
        from google.cloud import datastore
        entity = datastore.Entity(
            key=_model_provider_key(client, user_id, provider_id),
            exclude_from_indexes=("encrypted_secret", "wrapped_data_key"),
        )
        entity.update(record)
        client.put(entity)

    return ModelProviderStatus(
        provider_id=provider.id,
        name=provider.name,
        category=provider.category,
        description=provider.description,
        connected=True,
        default_model=selected_model,
        popular_models=provider.popular_models,
        default_base_url=provider.default_base_url,
        custom_base_url=selected_url if selected_url != provider.default_base_url else None,
        docs_url=provider.docs_url,
        updated_at=now,
        key_prefix=provider.key_prefix,
        masked_key=masked_key,
    )


def delete_user_model_provider(user_id: str, provider_id: str) -> None:
    client = _get_datastore_client()
    if client is None:
        _memory_user_providers.get(user_id, {}).pop(provider_id, None)
        return
    client.delete(_model_provider_key(client, user_id, provider_id))


def get_user_provider_status_list(user_id: str) -> List[ModelProviderStatus]:
    saved_map = {r["provider_id"]: r for r in list_user_model_providers(user_id)}
    results = []
    for pid, desc in CREDIBLE_PROVIDERS.items():
        rec = saved_map.get(pid)
        connected = bool(rec and rec.get("encrypted_secret"))
        custom_url = rec.get("custom_base_url") if rec else None
        results.append(
            ModelProviderStatus(
                provider_id=desc.id,
                name=desc.name,
                category=desc.category,
                description=desc.description,
                connected=connected,
                default_model=rec.get("default_model") if rec else desc.default_model,
                popular_models=desc.popular_models,
                default_base_url=desc.default_base_url,
                custom_base_url=custom_url if custom_url and custom_url != desc.default_base_url else None,
                docs_url=desc.docs_url,
                updated_at=rec.get("updated_at") if rec else None,
                key_prefix=desc.key_prefix,
                masked_key=rec.get("masked_key") if rec else None,
            )
        )
    return results


# ── Live Validation & Testing Helper ──────────────────────────────────────────

def verify_provider_credentials(
    provider_id: str,
    api_key: str,
    model_name: Optional[str] = None,
    custom_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    provider = CREDIBLE_PROVIDERS.get(provider_id)
    if not provider:
        return {"ok": False, "error": f"Unknown provider '{provider_id}'"}

    api_key = clean_http_header_value(api_key)
    if provider.requires_key and not api_key:
        return {"ok": False, "error": f"API key required for {provider.name}"}

    target_model = clean_provider_string(model_name) or provider.default_model
    target_url = clean_provider_string(custom_base_url) or provider.default_base_url

    t0 = time.perf_counter()
    try:
        provider_instance = create_chat_provider_for_config(
            provider_id=provider_id,
            api_key=api_key or "test-key",
            model_name=target_model,
            base_url=target_url,
        )
        reply = provider_instance.chat_completion(
            messages=[{"role": "user", "content": "Respond with 1 word: OK"}],
            temperature=0.0,
            max_tokens=5,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "message": f"Successfully connected to {provider.name}",
            "model": target_model,
            "sample_response": str(reply).strip()[:100],
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        err_msg = str(exc)
        return {
            "ok": False,
            "latency_ms": latency_ms,
            "error": err_msg[:300],
        }


# Alias for backward compatibility
test_provider_credentials = verify_provider_credentials


# ── Runtime ChatProvider Factory ──────────────────────────────────────────────

def create_chat_provider_for_config(
    provider_id: str,
    api_key: str,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> ChatProvider:
    provider = CREDIBLE_PROVIDERS.get(provider_id)
    target_model = clean_provider_string(model_name) or (provider.default_model if provider else "default")
    target_url = clean_provider_string(base_url) or (provider.default_base_url if provider else "")

    if provider_id == "anthropic":
        return AnthropicProvider(
            model_name=target_model,
            api_key=api_key,
            base_url=target_url or "https://api.anthropic.com/v1/messages",
        )
    elif provider_id == "huggingface":
        return HuggingFaceRouterProvider(
            model_name=target_model,
            api_key=api_key,
            base_url=target_url or "https://router.huggingface.co/v1/chat/completions",
        )
    elif provider_id == "openrouter":
        return OpenRouterProvider(
            model_name=target_model,
            api_key=api_key,
            base_url=target_url or "https://openrouter.ai/api/v1/chat/completions",
        )
    else:
        # Standard OpenAI-compatible (OpenAI, Google Gemini, DeepSeek, Qwen, Kimi, Meta, Groq, Mistral, xAI, Together, Perplexity, Cohere, Custom)
        return OpenAICompatibleProvider(
            model_name=target_model,
            api_key=api_key,
            base_url=target_url or "https://api.openai.com/v1/chat/completions",
        )


def get_user_chat_provider(user_id: str, provider_id: str, model_override: Optional[str] = None) -> Optional[ChatProvider]:
    rec = read_user_model_provider(user_id, provider_id)
    if not rec or not rec.get("encrypted_secret"):
        return None
    try:
        secret = _decrypt_provider_secret(user_id, provider_id, rec)
        api_key = secret.get("api_key", "")
        model_name = model_override or secret.get("default_model")
        base_url = secret.get("custom_base_url")
        return create_chat_provider_for_config(
            provider_id=provider_id,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
    except Exception:
        return None
