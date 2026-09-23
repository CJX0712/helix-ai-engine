"""Core package: configuration, types, interfaces, registry, errors, logging."""

from .config import Settings, settings
from .types import (
    Role,
    Message,
    Document,
    Chunk,
    ScoredChunk,
    EmbeddingVector,
)
from .interfaces import (
    LLMProvider,
    EmbeddingProvider,
    VectorStore,
    Retriever,
    Tool,
    Agent,
)
from .errors import (
    HelixError,
    ConfigurationError,
    ProviderError,
    ToolError,
    AgentError,
    RetrievalError,
)
from .registry import Container

__all__ = [
    "Settings",
    "settings",
    "Role",
    "Message",
    "Document",
    "Chunk",
    "ScoredChunk",
    "EmbeddingVector",
    "LLMProvider",
    "EmbeddingProvider",
    "VectorStore",
    "Retriever",
    "Tool",
    "Agent",
    "HelixError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "AgentError",
    "RetrievalError",
    "Container",
]
