"""Shared base helpers for LLM providers."""
from __future__ import annotations

from typing import List

from ...core.interfaces import LLMProvider
from ...core.types import Message


class BaseLLMProvider:
    """Common utilities for LLM adapters (payload conversion)."""

    name: str = "base-llm"

    @staticmethod
    def to_payload(messages: List[Message]) -> List[dict]:
        """Convert Helix messages to the OpenAI chat-completions payload shape."""
        out = []
        for m in messages:
            item = {"role": m.role.value, "content": m.content}
            if m.name:
                item["name"] = m.name
            out.append(item)
        return out

    async def stream(self, messages: List[Message], **kwargs):
        # Default streaming: emit the full answer as a single chunk.
        yield await self.generate(messages, **kwargs)

    async def close(self) -> None:
        return None
