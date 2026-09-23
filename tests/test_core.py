"""Tests for the core data types and error hierarchy."""
from helix.core.errors import (
    AgentError,
    ConfigurationError,
    HelixError,
    ProviderError,
    RetrievalError,
    ToolError,
)
from helix.core.types import Chunk, Message, Role, ScoredChunk


def test_role_values():
    assert Role.SYSTEM.value == "system"
    assert Role.TOOL.value == "tool"


def test_message_creation():
    m = Message(role=Role.USER, content="hi")
    assert m.role == Role.USER
    assert m.content == "hi"


def test_error_hierarchy():
    for cls in (ProviderError, ToolError, AgentError, RetrievalError, ConfigurationError):
        assert issubclass(cls, HelixError)


def test_scored_chunk_roundtrip():
    c = Chunk(id="c1", document_id="d1", text="t", index=0)
    sc = ScoredChunk(chunk=c, score=0.9)
    assert sc.score == 0.9
    assert sc.chunk.document_id == "d1"
