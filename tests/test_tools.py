"""Tests for atomic tools."""
import asyncio

import pytest

from helix.core.errors import ToolError
from helix.tools.calculator import CalculatorTool, safe_calculate
from helix.tools.text_tool import TextTool
from helix.tools.time_tool import TimeTool


def test_safe_calculate():
    assert safe_calculate("2 + 3 * 4") == 14
    assert safe_calculate("(10 - 2) / 4") == 2.0


def test_safe_calculate_rejects_names():
    with pytest.raises(ToolError):
        safe_calculate("__import__('os').system('echo hi')")


def test_tool_runners():
    async def run():
        assert "14" in await CalculatorTool().run("2 + 3 * 4")
        assert await TimeTool().run("")  # returns a non-empty timestamp
        out = await TextTool().run("upper: hello")
        assert out == "HELLO"

    asyncio.run(run())
