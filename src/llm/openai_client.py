from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from src.llm.base_client import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI-compatible chat/completions APIs."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("LLM_KEY")
        self._base_url = base_url or os.getenv("LLM_ENDPOINT")
        self._model = model

        if not self._api_key:
            raise ValueError("Missing LLM_KEY in environment.")
        if not self._base_url:
            raise ValueError("Missing LLM_ENDPOINT in environment.")

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def model_name(self) -> str:
        return self._model

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        resp = self._client.completions.create(
            model=self._model,
            prompt=prompt,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.text or "",
            model=self._model,
            prompt_tokens=(usage.prompt_tokens if usage else 0),
            completion_tokens=(usage.completion_tokens if usage else 0),
            finish_reason=(choice.finish_reason or "stop"),
            raw=resp,
        )

    def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = resp.usage

        content = choice.message.content
        if isinstance(content, list):
            # Some OpenAI-compatible providers may return chunked content parts.
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = content or ""

        return LLMResponse(
            text=text,
            model=self._model,
            prompt_tokens=(usage.prompt_tokens if usage else 0),
            completion_tokens=(usage.completion_tokens if usage else 0),
            finish_reason=(choice.finish_reason or "stop"),
            raw=resp,
        )
