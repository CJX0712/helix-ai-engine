"""Shared base for embedding providers."""
from __future__ import annotations


class BaseEmbeddingProvider:
    name: str = "base-embed"
    dimension: int = 0
