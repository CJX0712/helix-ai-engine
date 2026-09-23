"""Interface contracts (Protocols) for every pluggable component.

Modules depend ONLY on these protocols, never on concrete implementations.
This is what makes each module independently verifiable and hot-swappable.
"""
from __future__ import annotations

from typing import AsyncIterator, List, Optional, Protocol, runtime_checkable

from ..core.types import Chunk, Message, ScoredChunk


@runtime_checkable
class LLMProvider(Protocol):
    """Produces text from a sequence of chat messages."""

    name: str

    async def generate(self, messages: List[Message], **kwargs) -> str:
        """Return a single completion for the given conversation."""
        ...

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Yield completion tokens incrementally."""
        ...

    async def close(self) -> None:
        """Release underlying resources (e.g. HTTP connections)."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into fixed-dimensional vectors."""

    name: str
    dimension: int

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk vectors and retrieves the nearest neighbours."""

    name: str

    async def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Store chunks with their embedding vectors."""
        ...

    async def search(self, vector: List[float], top_k: int) -> List[ScoredChunk]:
        """Return the top_k most similar chunks to ``vector``."""
        ...

    async def count(self) -> int:
        """Return the number of stored chunks."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Resolves a natural-language query into scored chunks."""

    async def retrieve(self, query: str, top_k: int) -> List[ScoredChunk]:
        ...


@runtime_checkable
class Tool(Protocol):
    """An atomic, callable capability exposed to an agent."""

    name: str
    description: str

    async def run(self, input: str) -> str:
        """Execute the tool with the given input string."""
        ...


@runtime_checkable
class Agent(Protocol):
    """Drives a task to completion, possibly using tools and an LLM."""

    name: str

    async def run(self, task: str) -> str:
        """Execute ``task`` and return the final answer."""
        ...
