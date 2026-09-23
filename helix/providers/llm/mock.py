"""Deterministic, offline-safe mock LLM.

Used for tests, CI and environments without a reachable model endpoint. When a
``script`` is supplied (list of strings) responses are returned in order; this
lets agent tests drive a precise thought/action sequence. Otherwise a stable
echo-style response is returned.
"""
from __future__ import annotations

from typing import AsyncIterator, List, Optional

from ...core.types import Message


class MockLLMProvider:
    name = "mock-llm"

    def __init__(
        self,
        script: Optional[List[str]] = None,
        default_reply: Optional[str] = None,
    ) -> None:
        self._script = list(script) if script else None
        self._default = default_reply or "This is a deterministic mock response from Helix."
        self._counter = 0

    async def generate(self, messages: List[Message], **kwargs) -> str:
        self._counter += 1
        if self._script:
            if self._script:
                return self._script.pop(0)
        last = messages[-1].content if messages else ""
        return f"[mock:{self._counter}] acknowledging: {last[:140]}"

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        yield await self.generate(messages, **kwargs)

    async def close(self) -> None:
        return None
