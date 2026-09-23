"""Transparent fallback wrapper for LLM providers.

Tries the primary (real) provider first; on the first :class:`ProviderError`
it switches permanently to the fallback (deterministic mock) so the whole
pipeline keeps working even when no model endpoint is reachable.
"""
from __future__ import annotations

from typing import AsyncIterator, List

from ...core.errors import ProviderError
from ...core.logging import get_logger
from ...core.types import Message


class AutoFallbackLLM:
    name = "auto-fallback-llm"

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_fallback = False
        self._log = get_logger("helix.llm")

    async def generate(self, messages: List[Message], **kwargs) -> str:
        if self._use_fallback:
            return await self._fallback.generate(messages, **kwargs)
        try:
            return await self._primary.generate(messages, **kwargs)
        except ProviderError as exc:
            self._log.warning("Primary LLM unreachable (%s); using fallback.", exc)
            self._use_fallback = True
            return await self._fallback.generate(messages, **kwargs)

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        if self._use_fallback:
            async for tok in self._fallback.stream(messages, **kwargs):
                yield tok
            return
        try:
            async for tok in self._primary.stream(messages, **kwargs):
                yield tok
        except ProviderError as exc:
            self._log.warning("Primary LLM stream failed (%s); using fallback.", exc)
            self._use_fallback = True
            async for tok in self._fallback.stream(messages, **kwargs):
                yield tok

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()
