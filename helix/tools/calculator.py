"""Safe arithmetic calculator tool.

Uses Python's ``ast`` to evaluate only arithmetic expressions (no names, no
calls), so it is safe to expose to an agent without a sandbox escape.
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from ..core.errors import ToolError

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: Any) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ToolError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ToolError("Unsupported expression element.")


def safe_calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression: {exc}") from exc


class CalculatorTool:
    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '(12 + 8) * 3'."

    async def run(self, input: str) -> str:
        result = safe_calculate(input.strip())
        return f"{result}"
