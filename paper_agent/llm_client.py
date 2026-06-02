from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    temperature: float = 0.2
    max_tokens: int = 2048

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> "LLMConfig":
        resolved_model = model or os.getenv("VLLM_MODEL") or os.getenv("OPENAI_MODEL")
        if not resolved_model:
            raise ValueError(
                "No model was provided. Use --model or set VLLM_MODEL in your environment."
            )

        return cls(
            model=resolved_model,
            base_url=base_url
            or os.getenv("VLLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL,
            api_key=api_key
            or os.getenv("VLLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or DEFAULT_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class VLLMClient:
    """Small wrapper around vLLM's OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai is required to call the vLLM OpenAI-compatible API. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        request = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = self._client.chat.completions.create(
                **request,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except TypeError:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to call vLLM endpoint {self.config.base_url}. "
                "Make sure vLLM is running and the model name is correct."
            ) from exc

        content = response.choices[0].message.content
        return content.strip() if content else ""
