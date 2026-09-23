"""Transparent fallback wrapper for embedding providers."""
from __future__ import annotations

from typing import List

from ...core.errors import ProviderError
from ...core.logging import get_logger


class AutoFallbackEmbedding:
    name = "auto-fallback-embed"

    def __init__(self, primary, fallback) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_fallback = False
        self._log = get_logger("helix.embed")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if self._use_fallback:
            return await self._fallback.embed(texts)
        try:
            return await self._primary.embed(texts)
        except ProviderError as exc:
            self._log.warning("Primary embedding unreachable (%s); using fallback.", exc)
            self._use_fallback = True
            return await self._fallback.embed(texts)

    @property
    def dimension(self) -> int:
        return self._fallback.dimension

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()
