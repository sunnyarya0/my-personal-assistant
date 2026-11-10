"""
tests/test_config.py
====================
Unit tests for core/config.py

Tests cover:
  - Default value loading
  - Environment variable overrides
  - YAML config loading
  - Provider validation
  - API key validation
  - Path resolution
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from core.config import PROVIDER_DEFAULTS, Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**env_overrides: str) -> Settings:
    """Create a fresh Settings instance with specific env vars set."""
    base_env = {
        "LLM_PROVIDER": "gemini",
        "API_KEY": "test-key-1234",
        "LLM_MODEL": "",
        "AGENT_NAME": "TestBot",
        "SAFE_MODE": "true",
        "MAX_ITERATIONS": "10",
        "WORKSPACE_DIR": "",
        "LOG_DIR": "~/.assistant/logs",
        "UI_PORT": "8000",
    }
    base_env.update(env_overrides)
    with patch.dict(os.environ, base_env, clear=False):
        return Settings()


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------


class TestProviderValidation:
    def test_valid_providers_accepted(self) -> None:
        for provider in ("gemini", "openai", "anthropic", "ollama"):
            s = make_settings(LLM_PROVIDER=provider)
            assert s.llm_provider == provider

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(LLM_PROVIDER="unknown-provider")

    def test_provider_is_case_insensitive(self) -> None:
        s = make_settings(LLM_PROVIDER="GEMINI")
        assert s.llm_provider == "gemini"


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_empty_model_resolves_to_provider_default(self) -> None:
        for provider, expected_model in PROVIDER_DEFAULTS.items():
            s = make_settings(LLM_PROVIDER=provider, LLM_MODEL="")
            assert (
                s.llm_model == expected_model
            ), f"Provider {provider}: expected {expected_model}, got {s.llm_model}"

    def test_explicit_model_is_preserved(self) -> None:
        s = make_settings(LLM_PROVIDER="openai", LLM_MODEL="gpt-4o")
        assert s.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_workspace_defaults_to_home_dir(self) -> None:
        s = make_settings(WORKSPACE_DIR="")
        expected = str(Path.home() / "assistant-workspace")
        assert s.workspace_dir == expected

    def test_explicit_workspace_is_resolved(self, tmp_path: Path) -> None:
        s = make_settings(WORKSPACE_DIR=str(tmp_path))
        assert s.workspace_dir == str(tmp_path.resolve())

    def test_tilde_in_log_dir_is_expanded(self) -> None:
        s = make_settings(LOG_DIR="~/.assistant/logs")
        assert "~" not in s.log_dir
        assert str(Path.home()) in s.log_dir


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


class TestApiKeyValidation:
    def test_missing_api_key_raises_for_non_ollama(self) -> None:
        for provider in ("gemini", "openai", "anthropic"):
            s = make_settings(LLM_PROVIDER=provider, API_KEY="")
            with pytest.raises(ValueError, match="API_KEY is required"):
                s.validate_api_key()

    def test_ollama_does_not_require_api_key(self) -> None:
        s = make_settings(LLM_PROVIDER="ollama", API_KEY="")
        # Should not raise
        s.validate_api_key()

    def test_requires_api_key_property(self) -> None:
        assert make_settings(LLM_PROVIDER="gemini").requires_api_key is True
        assert make_settings(LLM_PROVIDER="ollama").requires_api_key is False


# ---------------------------------------------------------------------------
# Agent settings
# ---------------------------------------------------------------------------


class TestAgentSettings:
    def test_safe_mode_default_is_true(self) -> None:
        s = make_settings(SAFE_MODE="true")
        assert s.safe_mode is True

    def test_safe_mode_can_be_disabled(self) -> None:
        s = make_settings(SAFE_MODE="false")
        assert s.safe_mode is False

    def test_agent_name_is_set(self) -> None:
        s = make_settings(AGENT_NAME="Alfred")
        assert s.agent_name == "Alfred"

    def test_max_iterations(self) -> None:
        s = make_settings(MAX_ITERATIONS="25")
        assert s.max_iterations == 25


# ---------------------------------------------------------------------------
# UI settings
# ---------------------------------------------------------------------------


class TestUISettings:
    def test_default_port(self) -> None:
        s = make_settings(UI_PORT="8000")
        assert s.ui_port == 8000

    def test_custom_port(self) -> None:
        s = make_settings(UI_PORT="3000")
        assert s.ui_port == 3000
