"""Configuration loaded from environment / .env file.

Uses pydantic-settings so every value can be overridden via environment
variables prefixed with ``HELIX_`` (e.g. ``HELIX_LLM_MODEL``).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Engine-wide configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="HELIX_", extra="ignore"
    )

    # LLM provider (OpenAI-compatible endpoint, e.g. Ollama's /v1)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:0.5b"

    # Embedding provider (OpenAI-compatible endpoint)
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "ollama"
    embedding_model: str = "nomic-embed-text"

    # Provider resolution: auto | openai | mock
    #   auto  -> try the real provider, transparently fall back to a local
    #            deterministic provider if the endpoint is unreachable
    #   openai-> always use the real provider (errors surface)
    #   mock  -> always use the local deterministic provider (offline-safe)
    provider_mode: str = "auto"

    vectorstore_type: str = "memory"
    top_k: int = 4
    max_agent_iterations: int = 6
    log_level: str = "INFO"

    def require_mode(self) -> str:
        mode = self.provider_mode.lower()
        if mode not in {"auto", "openai", "mock"}:
            raise ValueError(f"Invalid provider_mode: {mode}")
        return mode


# Module-level singleton, lazily reads .env / environment on first import.
settings = Settings()
