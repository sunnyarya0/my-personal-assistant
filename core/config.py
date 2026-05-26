"""
core/config.py
==============
Loads, validates, and exposes all application configuration.

Priority order (highest → lowest):
  1. Environment variables
  2. .env file
  3. config/settings.yaml
  4. Built-in defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Supported LLM providers
# ---------------------------------------------------------------------------
LLMProvider = Literal["gemini", "openai", "anthropic", "ollama"]

# Default models per provider
PROVIDER_DEFAULTS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "llama3.2",
}

# Path to the YAML settings file
_SETTINGS_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


def _load_yaml_defaults() -> dict:
    """Load default values from config/settings.yaml."""
    if not _SETTINGS_PATH.exists():
        return {}
    with open(_SETTINGS_PATH) as f:
        data = yaml.safe_load(f) or {}

    # Flatten nested YAML into the flat structure that Settings expects
    flat: dict = {}
    llm = data.get("llm", {})
    agent = data.get("agent", {})
    storage = data.get("storage", {})
    ui = data.get("ui", {})
    security = data.get("security", {})

    flat["llm_provider"] = llm.get("provider", "gemini")
    flat["llm_model"] = llm.get("model", "")
    flat["llm_temperature"] = llm.get("temperature", 0.7)
    flat["llm_max_tokens"] = llm.get("max_tokens", 8192)
    flat["ollama_base_url"] = llm.get("ollama_base_url", "http://localhost:11434")

    flat["agent_name"] = agent.get("name", "Jarvis")
    flat["safe_mode"] = agent.get("safe_mode", True)
    flat["max_iterations"] = agent.get("max_iterations", 10)
    flat["verbose"] = agent.get("verbose", False)

    flat["workspace_dir"] = storage.get("workspace_dir", "")
    flat["log_dir"] = storage.get("log_dir", "~/.assistant/logs")
    flat["memory_dir"] = storage.get("memory_dir", "~/.assistant/memory")

    flat["ui_port"] = ui.get("port", 8000)
    flat["auto_open_browser"] = ui.get("auto_open_browser", True)

    flat["allowed_domains"] = security.get("allowed_domains", [])
    flat["rate_limit_rpm"] = security.get("rate_limit_rpm", 60)

    return flat


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Central configuration object for the Personal AI Agent Assistant.

    All fields can be set via environment variables (uppercase, no prefix)
    or via the .env file. YAML defaults are loaded as fallback values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: LLMProvider = Field(default="gemini", alias="LLM_PROVIDER")
    api_key: str = Field(default="", alias="API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=8192, gt=0)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # --- Agent ---
    agent_name: str = Field(default="Jarvis", alias="AGENT_NAME")
    safe_mode: bool = Field(default=True, alias="SAFE_MODE")
    max_iterations: int = Field(default=10, gt=0, alias="MAX_ITERATIONS")
    verbose: bool = Field(default=False)

    # --- Storage ---
    workspace_dir: str = Field(default="", alias="WORKSPACE_DIR")
    log_dir: str = Field(default="~/.assistant/logs", alias="LOG_DIR")
    memory_dir: str = Field(default="~/.assistant/memory")

    # --- Web UI ---
    ui_port: int = Field(default=8000, gt=0, lt=65536, alias="UI_PORT")
    auto_open_browser: bool = Field(default=True)

    # --- Security ---
    allowed_domains: list[str] = Field(default_factory=list, alias="ALLOWED_DOMAINS")
    rate_limit_rpm: int = Field(default=60, gt=0, alias="RATE_LIMIT_RPM")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        valid = {"gemini", "openai", "anthropic", "ollama"}
        v = str(v).lower().strip()
        if v not in valid:
            raise ValueError(f"LLM_PROVIDER must be one of: {', '.join(sorted(valid))}. Got: '{v}'")
        return v

    @model_validator(mode="after")
    def resolve_model_and_workspace(self) -> "Settings":
        # If no model specified, use provider default
        if not self.llm_model:
            self.llm_model = PROVIDER_DEFAULTS[self.llm_provider]

        # Resolve workspace_dir: default to ~/assistant-workspace
        if not self.workspace_dir:
            self.workspace_dir = str(Path.home() / "assistant-workspace")

        # Expand ~ in paths
        self.workspace_dir = str(Path(self.workspace_dir).expanduser().resolve())
        self.log_dir = str(Path(self.log_dir).expanduser().resolve())
        self.memory_dir = str(Path(self.memory_dir).expanduser().resolve())

        # Clean and normalize allowed_domains
        self.allowed_domains = [d.lower().strip() for d in self.allowed_domains if d.strip()]

        return self

    @property
    def requires_api_key(self) -> bool:
        """True if the chosen provider needs an API key."""
        return self.llm_provider != "ollama"

    def validate_api_key(self) -> None:
        """Raise ValueError if an API key is required but missing."""
        if self.requires_api_key and not self.api_key:
            raise ValueError(
                f"API_KEY is required for provider '{self.llm_provider}'. "
                f"Set it in your .env file or as an environment variable.\n"
                f"See .env.example for instructions."
            )

    def ensure_directories(self) -> None:
        """Create workspace, log, and memory directories if they don't exist."""
        for directory in [self.workspace_dir, self.log_dir, self.memory_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton — load once, use everywhere
# ---------------------------------------------------------------------------

def _build_settings() -> Settings:
    """Build Settings, applying YAML defaults as fallback."""
    yaml_defaults = _load_yaml_defaults()

    # Inject YAML values as env vars ONLY if the env var isn't already set.
    # This respects the priority: env > .env > yaml > built-in defaults.
    for key, value in yaml_defaults.items():
        env_key = key.upper()
        if env_key not in os.environ and value != "" and value is not None:
            os.environ.setdefault(env_key, str(value))

    return Settings()


settings: Settings = _build_settings()


__all__ = ["Settings", "settings", "LLMProvider", "PROVIDER_DEFAULTS"]
