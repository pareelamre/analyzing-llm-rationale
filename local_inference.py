"""Compatibility wrapper for local Qwen inference."""

from pathlib import Path
import sys
from typing import Dict, List, Optional

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analyzing_llm_rationale.providers import LocalQwenProvider


class LocalQwenInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        self._provider = LocalQwenProvider(model_name=model_name, device=device)
        self._provider._ensure_loaded()
        self.device = self._provider.resolved_device

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict:
        response = self._provider.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response,
                    }
                }
            ]
        }


_model_instance: Optional[LocalQwenInference] = None


def get_model() -> LocalQwenInference:
    global _model_instance
    if _model_instance is None:
        _model_instance = LocalQwenInference()
    return _model_instance


def chat_completion_local(messages: List[Dict[str, str]], temperature: float = 0.0, **kwargs) -> Dict:
    model = get_model()
    return model.chat_completion(messages, temperature, **kwargs)
