from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Structured response from any LLM client."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLMClient(ABC):
    """
    Abstract interface for any LLM client.

    Implement this class to connect a specific LLM provider
    (OpenAI, Anthropic, local Ollama, HuggingFace, etc.).

    Example
    -------
    class OpenAIClient(BaseLLMClient):
        def __init__(self, api_key: str, model: str = "gpt-4o"):
            self._client = openai.OpenAI(api_key=api_key)
            self._model = model

        @property
        def model_name(self) -> str:
            return self._model

        def complete(self, prompt: str, **kwargs) -> LLMResponse:
            resp = self._client.completions.create(
                model=self._model, prompt=prompt, **kwargs
            )
            return LLMResponse(
                text=resp.choices[0].text,
                model=self._model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                finish_reason=resp.choices[0].finish_reason,
                raw=resp,
            )

        def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages, **kwargs
            )
            return LLMResponse(
                text=resp.choices[0].message.content,
                model=self._model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                finish_reason=resp.choices[0].finish_reason,
                raw=resp,
            )
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. 'gpt-4o', 'llama-3-8b')."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Raw text completion.

        Parameters
        ----------
        prompt:
            The full prompt string.
        **kwargs:
            Provider-specific parameters (temperature, max_tokens, …).
        """

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """
        Chat-style completion.

        Parameters
        ----------
        messages:
            List of {"role": ..., "content": ...} dicts.
        **kwargs:
            Provider-specific parameters.
        """

    def ask(self, question: str, system: Optional[str] = None, **kwargs) -> str:
        """One-shot question → answer string (convenience wrapper over chat)."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})
        return self.chat(messages, **kwargs).text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
