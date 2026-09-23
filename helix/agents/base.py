"""ReAct prompt building and action parsing (model-agnostic).

The agent uses a textual action protocol (``ACTION:`` / ``INPUT:`` / ``FINAL:``)
rather than native function-calling so it works with ANY chat model, including
local Ollama models and the deterministic mock used in tests.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

_ACTION_RE = re.compile(
    r"ACTION:\s*(?P<name>[A-Za-z0-9_\-]+)\s*INPUT:\s*(?P<input>.*?)(?=\nACTION:|\nFINAL:|$)",
    re.DOTALL,
)
_FINAL_RE = re.compile(r"FINAL:\s*(?P<answer>.*)", re.DOTALL)


def build_system_prompt(tools: Dict[str, object]) -> str:
    lines = [
        "You are Helix, a reasoning agent. Solve the task step by step.",
        "You have access to these tools:",
    ]
    for name, tool in tools.items():
        desc = getattr(tool, "description", name)
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("Respond in EXACTLY one of these formats:")
    lines.append("ACTION: <tool_name>")
    lines.append("INPUT: <input for the tool>")
    lines.append("")
    lines.append("FINAL: <answer>")
    lines.append("")
    lines.append("Use ACTION to call a tool, then continue. When you have the final")
    lines.append("answer, respond with FINAL. Do not invent tool names.")
    return "\n".join(lines)


def parse_action(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (kind, tool_name_or_none, value).

    kind is either ``"final"`` or ``"action"``.
    """
    final = _FINAL_RE.search(text)
    if final:
        return "final", None, final.group("answer").strip()
    action = _ACTION_RE.search(text)
    if action:
        return "action", action.group("name").strip(), action.group("input").strip()
    # No recognised marker: treat the whole text as a final answer.
    return "final", None, text.strip()
