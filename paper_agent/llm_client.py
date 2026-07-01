from __future__ import annotations

import os
import time
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
    """Small wrapper around vLLM's OpenAI-compatible chat endpoint with retry."""

    _thinking_supported: bool | None = None

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
            import httpx
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai is required to call the vLLM OpenAI-compatible API. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.config = config
        http_client = httpx.Client(timeout=600, trust_env=False)
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=600,
            http_client=http_client,
        )

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> str:
        request = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                if VLLMClient._thinking_supported is not False:
                    response = self._client.chat.completions.create(
                        **request,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    if VLLMClient._thinking_supported is None:
                        VLLMClient._thinking_supported = True
                else:
                    response = self._client.chat.completions.create(**request)
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except TypeError:
                VLLMClient._thinking_supported = False
                continue
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"LLM call failed ({exc}), retrying in {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to call vLLM endpoint {self.config.base_url} after 3 attempts. "
            "Make sure vLLM is running and the model name is correct. "
            f"Last error: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc
