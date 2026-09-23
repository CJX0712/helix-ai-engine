"""OpenAI-compatible embedding adapter.

Works with any server exposing the OpenAI embeddings protocol, including
Ollama (``nomic-embed-text``), OpenAI, etc.
"""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

import httpx

from ...core.errors import ProviderError
from ...core.logging import get_logger
from .base import BaseEmbeddingProvider


def _is_local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("0.0.0.0")


class OpenAICompatibleEmbedding(BaseEmbeddingProvider):
    name = "openai-compatible-embed"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int = 0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=not _is_local(base_url))
        self._log = get_logger("helix.embed")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = await self._client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Embedding request failed: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"Malformed embedding response: {exc}") from exc

        ordered = sorted(data, key=lambda d: d["index"])
        vectors = [d["embedding"] for d in ordered]
        if self.dimension == 0 and vectors:
            self.dimension = len(vectors[0])
        return vectors

    async def close(self) -> None:
        await self._client.aclose()
