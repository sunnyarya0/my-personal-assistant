"""
tests/test_llm_client.py
========================
Unit tests for core/llm_client.py

All tests use mocked SDK responses — no live API calls are made.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import (
    AnthropicClient,
    GeminiClient,
    LLMResponse,
    OllamaClient,
    OpenAIClient,
    get_llm_client,
    ping_llm,
)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_total_tokens(self) -> None:
        r = LLMResponse(content="hello", model="m", provider="p", input_tokens=10, output_tokens=5)
        assert r.total_tokens == 15

    def test_defaults(self) -> None:
        r = LLMResponse(content="hi", model="gpt", provider="openai")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.raw is None


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class TestGeminiClient:
    def _make_client(self) -> GeminiClient:
        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.GenerationConfig = MagicMock(return_value={})
        # Patch sys.modules AND the google namespace so the import succeeds
        google_mock = MagicMock()
        google_mock.generativeai = mock_genai
        with patch.dict(
            "sys.modules",
            {"google": google_mock, "google.generativeai": mock_genai},
        ):
            client = GeminiClient(
                api_key="fake-key",
                model="gemini-2.0-flash",
                temperature=0.7,
                max_tokens=1024,
            )
        # Replace the internal client with a fresh mock for assertion
        client._client = mock_model
        return client

    def test_chat_returns_llm_response(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_response.usage_metadata = SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5
        )
        client._client.generate_content.return_value = mock_response

        result = client.chat("Say hello")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Gemini!"
        assert result.provider == "gemini"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_chat_with_system_prompt(self) -> None:
        client = self._make_client()
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.usage_metadata = None
        client._client.generate_content.return_value = mock_response

        client.chat("Hello", system_prompt="You are a helpful assistant.")

        call_args = client._client.generate_content.call_args[0][0]
        assert "You are a helpful assistant." in call_args
        assert "Hello" in call_args


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

class TestOpenAIClient:
    def _make_client(self) -> OpenAIClient:
        mock_openai_module = MagicMock()
        mock_openai_class = MagicMock()
        mock_openai_module.OpenAI = mock_openai_class

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            client = OpenAIClient(
                api_key="fake-key",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1024,
            )
        return client

    def test_chat_returns_llm_response(self) -> None:
        client = self._make_client()

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from OpenAI!"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 8
        mock_usage.completion_tokens = 4
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]
        mock_api_response.usage = mock_usage
        client._client.chat.completions.create.return_value = mock_api_response

        result = client.chat("Say hello")

        assert result.content == "Hello from OpenAI!"
        assert result.provider == "openai"
        assert result.input_tokens == 8
        assert result.output_tokens == 4


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class TestAnthropicClient:
    def _make_client(self) -> AnthropicClient:
        mock_anthropic_module = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            client = AnthropicClient(
                api_key="fake-key",
                model="claude-3-5-haiku-20241022",
                temperature=0.7,
                max_tokens=1024,
            )
        return client

    def test_chat_returns_llm_response(self) -> None:
        client = self._make_client()

        mock_block = MagicMock()
        mock_block.text = "Hello from Claude!"
        mock_usage = MagicMock()
        mock_usage.input_tokens = 12
        mock_usage.output_tokens = 6
        mock_api_response = MagicMock()
        mock_api_response.content = [mock_block]
        mock_api_response.usage = mock_usage
        client._client.messages.create.return_value = mock_api_response

        result = client.chat("Say hello")

        assert result.content == "Hello from Claude!"
        assert result.provider == "anthropic"
        assert result.input_tokens == 12
        assert result.output_tokens == 6

    def test_system_prompt_passed_as_kwarg(self) -> None:
        client = self._make_client()
        mock_block = MagicMock()
        mock_block.text = "OK"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)
        client._client.messages.create.return_value = mock_response

        client.chat("Hi", system_prompt="Be concise.")

        call_kwargs = client._client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "Be concise."


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def _make_client(self) -> OllamaClient:
        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            client = OllamaClient(
                base_url="http://localhost:11434",
                model="llama3.2",
                temperature=0.7,
                max_tokens=1024,
            )
        return client

    def test_chat_returns_llm_response(self) -> None:
        client = self._make_client()
        mock_response = {"message": {"content": "Hello from Ollama!"}}
        client._ollama.chat.return_value = mock_response

        result = client.chat("Say hello")

        assert result.content == "Hello from Ollama!"
        assert result.provider == "ollama"


# ---------------------------------------------------------------------------
# get_llm_client factory
# ---------------------------------------------------------------------------

class TestGetLLMClient:
    def test_factory_returns_correct_client_type(self) -> None:
        mock_genai = MagicMock()
        mock_genai.GenerationConfig = MagicMock(return_value={})
        google_mock = MagicMock()
        google_mock.generativeai = mock_genai
        with patch.dict(
            "sys.modules",
            {"google": google_mock, "google.generativeai": mock_genai},
        ):
            client = get_llm_client(provider="gemini", api_key="fake")
        assert isinstance(client, GeminiClient)

    def test_factory_raises_for_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client(provider="unknown")  # type: ignore


# ---------------------------------------------------------------------------
# ping_llm
# ---------------------------------------------------------------------------

class TestPingLLM:
    def test_ping_returns_false_on_missing_api_key(self) -> None:
        with patch("core.llm_client.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"
            mock_settings.requires_api_key = True
            mock_settings.validate_api_key.side_effect = ValueError("API_KEY is required")
            result = ping_llm()
        assert result is False

    def test_ping_returns_true_on_success(self) -> None:
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch("core.llm_client.settings") as mock_settings, \
             patch("core.llm_client.get_llm_client", return_value=mock_client):
            mock_settings.llm_provider = "gemini"
            mock_settings.llm_model = "gemini-2.0-flash"
            mock_settings.validate_api_key = MagicMock()
            result = ping_llm()

        assert result is True
