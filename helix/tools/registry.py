"""Tool registry / factory."""
from __future__ import annotations

from typing import Dict

from .calculator import CalculatorTool
from .time_tool import TimeTool
from .text_tool import TextTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, object] = {}

    def register(self, tool: object) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> object:
        return self._tools[name]

    def all(self) -> Dict[str, object]:
        return dict(self._tools)


def build_default_tools() -> Dict[str, object]:
    """Return the standard tool set used by the default agent."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(TextTool())
    return registry.all()
