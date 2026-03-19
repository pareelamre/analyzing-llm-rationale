from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

CONTEXT_WINDOW_SENTINEL = 1_000_000


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class RetryableProviderError(ProviderError):
    """The request may succeed if retried."""


class ContextLimitError(ProviderError):
    """The prompt exceeded the provider's effective context window."""


class ProviderResponseError(ProviderError):
    """The provider returned a non-retryable error."""


class ChatProvider:
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        ) -> str:
        raise NotImplementedError


def resolve_context_window(
    configured_limit: Optional[int],
    model_limit: Optional[int],
    tokenizer_limit: Optional[int],
) -> Optional[int]:
    for limit in (configured_limit, model_limit, tokenizer_limit):
        if isinstance(limit, int) and 0 < limit < CONTEXT_WINDOW_SENTINEL:
            return limit
    return None


def ensure_prompt_fits_context(
    input_tokens: int,
    max_tokens: int,
    context_window: Optional[int],
) -> None:
    if context_window is None:
        return
    if input_tokens + max_tokens > context_window:
        raise ContextLimitError(
            f"Prompt exceeds context window: input_tokens={input_tokens}, "
            f"max_tokens={max_tokens}, context_window={context_window}"
        )


@dataclass
class OpenAICompatibleProvider(ChatProvider):
    model_name: str
    api_key: str
    request_timeout_s: float = 120.0
    base_url: str = "https://api.openai.com/v1/chat/completions"
    missing_api_key_message: str = "API key must be set."
    def __post_init__(self) -> None:
        import requests

        if not self.api_key:
            raise ValueError(self.missing_api_key_message)

        self._requests = requests
        self._session = requests.Session()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": self.model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._session.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.request_timeout_s,
        )
        response_text = response.text[:500]
        if response.status_code == 400 and "maximum context length" in response.text.lower():
            raise ContextLimitError(response_text)
        if response.status_code in (408, 409, 425, 429) or response.status_code >= 500:
            raise RetryableProviderError(
                f"status={response.status_code} body={response_text}"
            )
        if response.status_code != 200:
            raise ProviderResponseError(
                f"status={response.status_code} body={response_text}"
            )

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError(f"Malformed provider response: {exc}") from exc


@dataclass
class HuggingFaceRouterProvider(OpenAICompatibleProvider):
    base_url: str = "https://router.huggingface.co/v1/chat/completions"
    missing_api_key_message: str = "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN must be set for hf-router."


@dataclass
class LocalQwenProvider(ChatProvider):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    torch_dtype: Optional[str] = None
    context_window: Optional[int] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._resolved_device = self.device

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "auto":
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.device == "cuda" and not torch.cuda.is_available():
            self._resolved_device = "cpu"
        else:
            self._resolved_device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if self._resolved_device == "cuda":
            dtype = torch.float16 if self.torch_dtype is None else getattr(torch, self.torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            dtype = torch.float32 if self.torch_dtype is None else getattr(torch, self.torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            model.to(self._resolved_device)

        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    @property
    def resolved_device(self) -> str:
        self._ensure_loaded()
        return self._resolved_device

    @property
    def effective_context_window(self) -> Optional[int]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        return resolve_context_window(
            configured_limit=self.context_window,
            model_limit=getattr(self._model.config, "max_position_embeddings", None),
            tokenizer_limit=getattr(self._tokenizer, "model_max_length", None),
        )

    def _generation_kwargs(self, temperature: float, max_tokens: int) -> Dict[str, object]:
        assert self._model is not None
        assert self._tokenizer is not None

        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is None:
            if temperature == 0.0:
                generation_kwargs["do_sample"] = False
            else:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = temperature
            return generation_kwargs

        generation_config = deepcopy(generation_config)
        generation_config.max_new_tokens = max_tokens
        generation_config.pad_token_id = self._tokenizer.pad_token_id
        generation_config.eos_token_id = self._tokenizer.eos_token_id

        if temperature == 0.0:
            generation_config.do_sample = False
            for field_name, neutral_value in (
                ("temperature", 1.0),
                ("top_p", 1.0),
                ("min_p", None),
                ("typical_p", 1.0),
                ("top_k", 50),
                ("epsilon_cutoff", 0.0),
                ("eta_cutoff", 0.0),
            ):
                if hasattr(generation_config, field_name):
                    setattr(generation_config, field_name, neutral_value)
        else:
            generation_config.do_sample = True
            generation_config.temperature = temperature

        generation_kwargs["generation_config"] = generation_config
        return generation_kwargs

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        self._ensure_loaded()

        import torch

        assert self._model is not None
        assert self._tokenizer is not None

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = self._tokenizer(text, return_tensors="pt")
        input_tokens = int(tokenized.input_ids.shape[1])
        ensure_prompt_fits_context(
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            context_window=self.effective_context_window,
        )
        inputs = tokenized.to(self._resolved_device)

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **self._generation_kwargs(temperature=temperature, max_tokens=max_tokens),
                )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "context" in message:
                raise ContextLimitError(str(exc)) from exc
            raise ProviderResponseError(str(exc)) from exc

        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)


def download_model_snapshot(model_name: str, cache_dir: Optional[Path] = None) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        resume_download=True,
    )
    AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    return Path(model_path)
