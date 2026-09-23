"""Tests for the ReAct agent (tool-use loop + graceful degradation)."""
import asyncio

from helix.core.config import Settings
from helix.core.registry import Container
from helix.providers.llm.mock import MockLLMProvider


def _container_with_script(script):
    c = Container(Settings(_env_file=None, provider_mode="mock"))
    c.llm = MockLLMProvider(script=script)
    c.agent = type(c.agent)(c.llm, c.tools, c.agent.max_iterations)
    return c


def test_agent_uses_tool():
    async def run():
        c = _container_with_script(
            [
                "ACTION: calculator\nINPUT: 2 + 3 * 4",
                "FINAL: The answer is 14.",
            ]
        )
        result = await c.agent.run("compute something")
        assert "14" in result

    asyncio.run(run())


def test_agent_unknown_tool_does_not_crash():
    async def run():
        c = _container_with_script(
            [
                "ACTION: nonexistent_tool\nINPUT: x",
                "FINAL: recovered",
            ]
        )
        result = await c.agent.run("task")
        assert result  # agent surfaces the final answer, no exception

    asyncio.run(run())
