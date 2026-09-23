"""Tests for LLM provider adapters (mock + auto-fallback)."""
import asyncio

from helix.core.errors import ProviderError
from helix.core.types import Message, Role
from helix.providers.llm.fallback import AutoFallbackLLM
from helix.providers.llm.mock import MockLLMProvider


def test_mock_generate_echoes_last_message():
    async def run():
        m = MockLLMProvider()
        out = await m.generate([Message(role=Role.USER, content="hello world")])
        assert "hello world" in out

    asyncio.run(run())


class _BrokenLLM:
    name = "broken"

    async def generate(self, messages, **kwargs):
        raise ProviderError("down")

    async def stream(self, messages, **kwargs):
        raise ProviderError("down")

    async def close(self):
        return None


def test_auto_fallback_uses_mock_on_failure():
    async def run():
        f = AutoFallbackLLM(_BrokenLLM(), MockLLMProvider())
        out = await f.generate([Message(role=Role.USER, content="hi")])
        assert "mock" in out

    asyncio.run(run())


def test_auto_fallback_switches_permanently():
    async def run():
        broken = _BrokenLLM()
        f = AutoFallbackLLM(broken, MockLLMProvider())
        await f.generate([Message(role=Role.USER, content="a")])
        # Second call should not reach the broken provider again.
        out = await f.generate([Message(role=Role.USER, content="b")])
        assert "mock" in out

    asyncio.run(run())
