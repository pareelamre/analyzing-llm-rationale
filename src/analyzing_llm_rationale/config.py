from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class VariantConfig:
    name: str
    prompt_path: str
    output_fields: Tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    result_label: str
    provider: str
    local_model_name: str
    router_model_name: str
    api_base_url: str | None = None
    api_key_env_var: str | None = None
    api_key_file: str | None = None
    max_tokens_cap: int | None = None
    context_window_tokens: int | None = None
    request_timeout_cap_s: float | None = None
    forecasting_enabled: bool = True
    chat_interface_enabled: bool = False
    track_record_enabled: bool = True
    fallback_model_chain: Tuple[str, ...] = ()


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_variant_configs(path: Path) -> Dict[str, VariantConfig]:
    data = load_yaml(path)
    raw_variants = data.get("variants")
    if not isinstance(raw_variants, dict):
        raise ValueError(f"Expected 'variants' mapping in {path}")

    variants: Dict[str, VariantConfig] = {}
    for name, payload in raw_variants.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Variant '{name}' must be a mapping")
        prompt_path = payload.get("prompt_path")
        output_fields = payload.get("output_fields")
        if not isinstance(prompt_path, str):
            raise ValueError(f"Variant '{name}' is missing string prompt_path")
        if not isinstance(output_fields, Sequence) or isinstance(output_fields, (str, bytes)):
            raise ValueError(f"Variant '{name}' is missing list output_fields")
        fields = tuple(str(field) for field in output_fields)
        variants[name] = VariantConfig(name=name, prompt_path=prompt_path, output_fields=fields)
    return variants


def load_model_configs(path: Path) -> Dict[str, ModelConfig]:
    data = load_yaml(path)
    raw_models = data.get("models")
    if not isinstance(raw_models, dict):
        raise ValueError(f"Expected 'models' mapping in {path}")

    models: Dict[str, ModelConfig] = {}
    for name, payload in raw_models.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Model '{name}' must be a mapping")
        result_label = payload.get("result_label")
        provider = payload.get("provider")
        local_model_name = payload.get("local_model_name")
        router_model_name = payload.get("router_model_name")
        api_base_url = payload.get("api_base_url")
        api_key_env_var = payload.get("api_key_env_var")
        api_key_file = payload.get("api_key_file")
        max_tokens_cap = payload.get("max_tokens_cap")
        context_window_tokens = payload.get("context_window_tokens")
        request_timeout_cap_s = payload.get("request_timeout_cap_s")
        forecasting_enabled = payload.get("forecasting_enabled", True)
        chat_interface_enabled = payload.get("chat_interface_enabled", False)
        track_record_enabled = payload.get("track_record_enabled", True)
        fallback_model_chain = payload.get("fallback_model_chain", [])
        if not all(isinstance(value, str) for value in (result_label, provider, local_model_name, router_model_name)):
            raise ValueError(
                f"Model '{name}' must define string result_label, provider, local_model_name, and router_model_name"
            )
        if api_base_url is not None and not isinstance(api_base_url, str):
            raise ValueError(f"Model '{name}' api_base_url must be a string when provided")
        if api_key_env_var is not None and not isinstance(api_key_env_var, str):
            raise ValueError(f"Model '{name}' api_key_env_var must be a string when provided")
        if api_key_file is not None and not isinstance(api_key_file, str):
            raise ValueError(f"Model '{name}' api_key_file must be a string when provided")
        if max_tokens_cap is not None and not isinstance(max_tokens_cap, int):
            raise ValueError(f"Model '{name}' max_tokens_cap must be an integer when provided")
        if isinstance(max_tokens_cap, int) and max_tokens_cap <= 0:
            raise ValueError(f"Model '{name}' max_tokens_cap must be positive when provided")
        if context_window_tokens is not None and not isinstance(context_window_tokens, int):
            raise ValueError(f"Model '{name}' context_window_tokens must be an integer when provided")
        if isinstance(context_window_tokens, int) and context_window_tokens <= 0:
            raise ValueError(f"Model '{name}' context_window_tokens must be positive when provided")
        if request_timeout_cap_s is not None and not isinstance(request_timeout_cap_s, (int, float)):
            raise ValueError(f"Model '{name}' request_timeout_cap_s must be numeric when provided")
        if isinstance(request_timeout_cap_s, (int, float)) and request_timeout_cap_s <= 0:
            raise ValueError(f"Model '{name}' request_timeout_cap_s must be positive when provided")
        if not isinstance(forecasting_enabled, bool):
            raise ValueError(f"Model '{name}' forecasting_enabled must be boolean when provided")
        if not isinstance(chat_interface_enabled, bool):
            raise ValueError(f"Model '{name}' chat_interface_enabled must be boolean when provided")
        if not isinstance(track_record_enabled, bool):
            raise ValueError(f"Model '{name}' track_record_enabled must be boolean when provided")
        if not isinstance(fallback_model_chain, Sequence) or isinstance(fallback_model_chain, (str, bytes)):
            raise ValueError(f"Model '{name}' fallback_model_chain must be a list when provided")
        fallback_chain = tuple(str(model) for model in fallback_model_chain)
        models[name] = ModelConfig(
            name=name,
            result_label=result_label,
            provider=provider,
            local_model_name=local_model_name,
            router_model_name=router_model_name,
            api_base_url=api_base_url,
            api_key_env_var=api_key_env_var,
            api_key_file=api_key_file,
            max_tokens_cap=max_tokens_cap,
            context_window_tokens=context_window_tokens,
            request_timeout_cap_s=(
                float(request_timeout_cap_s) if request_timeout_cap_s is not None else None
            ),
            forecasting_enabled=forecasting_enabled,
            chat_interface_enabled=chat_interface_enabled,
            track_record_enabled=track_record_enabled,
            fallback_model_chain=fallback_chain,
        )
    return models


def scads_hosted_model_allowlist(path: Path) -> Dict[str, str]:
    """Configured SCADS-hosted model labels mapped to provider model names."""
    models = load_model_configs(path)
    out: Dict[str, str] = {}
    for name, cfg in models.items():
        if (
            cfg.provider == "openai-compatible"
            and cfg.forecasting_enabled
            and cfg.api_key_env_var == "SCADS_AI_API_KEY"
            and cfg.api_base_url
            and "llm.scads.ai" in cfg.api_base_url
        ):
            out[name] = cfg.router_model_name
    return out


def scads_track_model_labels(path: Path) -> Tuple[str, ...]:
    """Default non-synthetic model labels for the track-record comparison board.

    A model can be `forecasting_enabled` (usable via /predict, the council
    ensemble, and chat) without being tracked on the live MTM/edge-board
    comparison -- e.g. one the tracking CI matrix no longer schedules, or one
    whose SCADS API key currently lacks access (`track_record_enabled:
    false`). Without this, such a model's paper-trading account sits frozen
    at the starting balance forever instead of being excluded from the board.
    """
    models = load_model_configs(path)
    return tuple(
        name for name in scads_hosted_model_allowlist(path)
        if models[name].track_record_enabled
    )


def scads_chat_model_options(path: Path) -> Tuple[ModelConfig, ...]:
    """SCADS-hosted text chat models exposed in the interactive composer."""
    return tuple(
        cfg
        for cfg in load_model_configs(path).values()
        if (
            cfg.chat_interface_enabled
            and cfg.provider == "openai-compatible"
            and cfg.forecasting_enabled
            and cfg.api_key_env_var == "SCADS_AI_API_KEY"
            and cfg.api_base_url
            and "llm.scads.ai" in cfg.api_base_url
        )
    )


def scads_hosted_model_fallbacks(path: Path) -> Dict[str, Tuple[str, ...]]:
    """Configured SCADS-hosted model labels mapped to provider fallback chains."""
    return {
        cfg.name: cfg.fallback_model_chain
        for cfg in scads_chat_model_options(path)
        if cfg.fallback_model_chain
    }


def temperature_to_tag(temperature: float) -> str:
    normalized = f"{temperature:.3f}".rstrip("0").rstrip(".")
    if not normalized:
        normalized = "0"
    return f"temperature_{normalized.replace('.', '')}"
