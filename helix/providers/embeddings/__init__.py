"""Embedding provider adapters."""
from .base import BaseEmbeddingProvider
from .openai_compatible import OpenAICompatibleEmbedding
from .deterministic import DeterministicEmbedding
from .fallback import AutoFallbackEmbedding

__all__ = [
    "BaseEmbeddingProvider",
    "OpenAICompatibleEmbedding",
    "DeterministicEmbedding",
    "AutoFallbackEmbedding",
]
