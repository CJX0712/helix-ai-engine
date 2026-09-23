"""Shared data models for the Helix engine.

All cross-module data contracts are defined here as Pydantic models so that
every module (LLM, embeddings, vector store, RAG, agents, API) speaks the same
typed language.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Conversation role for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single chat message exchanged with an LLM provider."""

    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class Document(BaseModel):
    """A raw document to be ingested into the RAG pipeline."""

    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk produced by splitting a document; the unit stored in a vector store."""

    id: str
    document_id: str
    text: str
    index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A retrieved chunk together with its similarity score."""

    chunk: Chunk
    score: float


class EmbeddingVector(BaseModel):
    """A single embedding vector."""

    vector: List[float]
