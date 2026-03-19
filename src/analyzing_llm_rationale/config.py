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
        models[name] = ModelConfig(
            name=name,
            result_label=result_label,
            provider=provider,
            local_model_name=local_model_name,
            router_model_name=router_model_name,
            api_base_url=api_base_url,
            api_key_env_var=api_key_env_var,
            api_key_file=api_key_file,
        )
    return models


def temperature_to_tag(temperature: float) -> str:
    normalized = f"{temperature:.3f}".rstrip("0").rstrip(".")
    if not normalized:
        normalized = "0"
    return f"temperature_{normalized.replace('.', '')}"
