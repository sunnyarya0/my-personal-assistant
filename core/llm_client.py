"""
core/llm_client.py
==================
Unified LLM client abstraction.

Provides a single `LLMClient` interface over all supported providers:
  - Google Gemini
  - OpenAI
  - Anthropic Claude
  - Ollama (local models)

Usage:
    from core.llm_client import get_llm_client

    client = get_llm_client()
    response = client.chat("Tell me a joke")
    print(response.content)
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from core.config import LLMProvider, settings

console = Console()


# ---------------------------------------------------------------------------
# Response dataclass — provider-agnostic output format
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Standardised response object returned by all LLM clients."""
    content: str                          # The text response from the model
    model: str                            # Model that generated the response
    provider: str                         # Provider that was used
    input_tokens: int = 0                 # Approximate input token count
    output_tokens: int = 0                # Approximate output token count
    raw: Any = field(default=None, repr=False)  # Raw provider response (for debugging)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseLLMClient(abc.ABC):
    """
    Abstract interface that all LLM provider clients must implement.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """
        Send a user prompt (and optional system prompt) to the model,
        wrapped with exponential backoff retry logic for resilience.
        """
        import random
        max_retries = 3
        initial_delay = 1.0
        
        for attempt in range(1, max_retries + 2):
            try:
                return self._chat_internal(prompt, system_prompt=system_prompt)
            except Exception as exc:
                if attempt > max_retries:
                    raise exc
                
                err_str = str(exc).lower()
                # Block repeating known credential errors
                if "401" in err_str or "403" in err_str or "api key" in err_str or "credentials" in err_str:
                    raise exc
                
                delay = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                console.print(
                    f"[yellow]⚠️  LLM call failed (attempt {attempt}/{max_retries + 1}): {exc}. "
                    f"Retrying in {delay:.2f}s...[/yellow]"
                )
                time.sleep(delay)

    @abc.abstractmethod
    def _chat_internal(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        """
        Abstract method implemented by subclasses to perform the actual API request.
        """

    def ping(self) -> bool:
        """
        Send a minimal prompt to verify connectivity and credentials.

        Returns:
            True if the provider responded correctly, False otherwise.
        """
        try:
            response = self.chat("Reply with exactly one word: ready")
            return bool(response.content.strip())
        except Exception as exc:
            console.print(f"[red]Ping failed:[/red] {exc}")
            return False


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """Google Gemini via google-generativeai SDK."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            self._genai = genai
            self._client = genai.GenerativeModel(
                model_name=model,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

    def _chat_internal(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
        response = self._client.generate_content(full_prompt)
        content = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=content,
            model=self.model,
            provider="gemini",
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            raw=response,
        )


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """OpenAI GPT models via official openai SDK."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Run: pip install openai"
            ) from exc

    def _chat_internal(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude models via official anthropic SDK."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "anthropic is not installed. Run: pip install anthropic"
            ) from exc

    def _chat_internal(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        content = response.content[0].text if response.content else ""
        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """Local models via Ollama (no API key required)."""

    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int) -> None:
        super().__init__(model, temperature, max_tokens)
        try:
            import ollama  # type: ignore
            self._ollama = ollama
            self._base_url = base_url
        except ImportError as exc:
            raise ImportError(
                "ollama is not installed. Run: pip install ollama"
            ) from exc

    def _chat_internal(self, prompt: str, system_prompt: str = "") -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        content = response["message"]["content"] if isinstance(response, dict) else response.message.content
        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            raw=response,
        )


# ---------------------------------------------------------------------------
# Factory function — the main public API
# ---------------------------------------------------------------------------

def get_llm_client(
    provider: LLMProvider | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseLLMClient:
    """
    Create and return the appropriate LLM client based on configuration.

    All arguments are optional — defaults are taken from `settings`.

    Args:
        provider: Override the LLM provider.
        api_key: Override the API key.
        model: Override the model name.
        temperature: Override sampling temperature.
        max_tokens: Override maximum output tokens.

    Returns:
        A configured LLM client instance.

    Raises:
        ValueError: If the provider is unknown or API key is missing.
    """
    _provider = provider or settings.llm_provider
    _api_key = api_key or settings.api_key
    _model = model or settings.llm_model
    _temperature = temperature if temperature is not None else settings.llm_temperature
    _max_tokens = max_tokens or settings.llm_max_tokens

    if _provider == "gemini":
        return GeminiClient(_api_key, _model, _temperature, _max_tokens)
    elif _provider == "openai":
        return OpenAIClient(_api_key, _model, _temperature, _max_tokens)
    elif _provider == "anthropic":
        return AnthropicClient(_api_key, _model, _temperature, _max_tokens)
    elif _provider == "ollama":
        return OllamaClient(settings.ollama_base_url, _model, _temperature, _max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: '{_provider}'")


def ping_llm() -> bool:
    """
    Quick connectivity check — verifies the configured LLM provider is reachable.

    Returns:
        True if ping succeeded, False otherwise.
    """
    try:
        settings.validate_api_key()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return False

    console.print(f"[cyan]Pinging {settings.llm_provider} ({settings.llm_model})...[/cyan]")
    start = time.perf_counter()
    client = get_llm_client()
    ok = client.ping()
    elapsed = time.perf_counter() - start

    if ok:
        console.print(f"[green]✓ LLM provider is reachable[/green] ({elapsed:.2f}s)")
    else:
        console.print(f"[red]✗ LLM provider did not respond correctly[/red]")
    return ok


__all__ = [
    "BaseLLMClient",
    "GeminiClient",
    "OpenAIClient",
    "AnthropicClient",
    "OllamaClient",
    "LLMResponse",
    "get_llm_client",
    "ping_llm",
]
