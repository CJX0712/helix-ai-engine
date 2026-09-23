"""OpenAI-compatible LLM adapter.

Works with any server exposing the OpenAI chat-completions protocol, including
Ollama (``http://localhost:11434/v1``), OpenAI, vLLM, LM Studio, etc. Reuses
the open-source ``httpx`` HTTP stack instead of vendoring a client.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, List, Optional
from urllib.parse import urlparse

import httpx

from ...core.errors import ProviderError
from ...core.logging import get_logger
from ...core.types import Message
from .base import BaseLLMProvider


def _is_local(url: str) -> bool:
    """Return True for loopback endpoints so we skip any HTTP proxy."""
    host = urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("0.0.0.0")


class OpenAICompatibleLLM(BaseLLMProvider):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # Loopback endpoints (e.g. local Ollama) must bypass any HTTP proxy.
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=not _is_local(base_url))
        self._log = get_logger("helix.llm")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, messages: List[Message], **kwargs) -> str:
        payload = {
            "model": self.model,
            "messages": self.to_payload(messages),
            "stream": False,
            **kwargs,
        }
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as exc:  # network / HTTP errors
            raise ProviderError(f"LLM request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"Malformed LLM response: {exc}") from exc

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": self.to_payload(messages),
            "stream": True,
            **kwargs,
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:") :].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    delta = data["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(f"LLM stream failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()
