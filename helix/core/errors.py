"""Unified error hierarchy for the Helix engine."""
from __future__ import annotations


class HelixError(Exception):
    """Base class for all Helix errors."""


class ConfigurationError(HelixError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(HelixError):
    """Raised when an upstream model/embedding provider fails."""


class ToolError(HelixError):
    """Raised when a tool execution fails."""


class AgentError(HelixError):
    """Raised when agent orchestration fails."""


class RetrievalError(HelixError):
    """Raised when retrieval from a vector store fails."""
