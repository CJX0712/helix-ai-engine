"""ReAct-style agent with pluggable LLM and tools."""
from __future__ import annotations

from typing import Dict, List

from ..core.errors import AgentError
from ..core.logging import get_logger
from ..core.types import Message, Role
from .base import build_system_prompt, parse_action


class ReActAgent:
    name = "react-agent"

    def __init__(self, llm, tools: Dict[str, object], max_iterations: int = 6) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self._log = get_logger("helix.agent")

    async def run(self, task: str) -> str:
        system = build_system_prompt(self.tools)
        history: List[Message] = [
            Message(role=Role.USER, content=f"Task: {task}")
        ]
        transcript = ""

        for step in range(self.max_iterations):
            messages = [Message(role=Role.SYSTEM, content=system)] + history
            try:
                reply = await self.llm.generate(messages)
            except Exception as exc:  # propagate provider failures clearly
                raise AgentError(f"LLM call failed at step {step}: {exc}") from exc

            transcript += f"\n[step {step}] {reply}"
            kind, tool_name, value = parse_action(reply)

            if kind == "final":
                return value or reply.strip()

            if kind == "action":
                tool = self.tools.get(tool_name)
                if tool is None:
                    observation = f"Error: unknown tool '{tool_name}'."
                else:
                    try:
                        observation = await tool.run(value or "")
                    except Exception as exc:  # tool failure should not crash agent
                        observation = f"Error running {tool_name}: {exc}"
                self._log.debug("tool %s -> %s", tool_name, observation[:80])
                history.append(
                    Message(role=Role.USER, content=f"Observation: {observation}")
                )
                continue

        raise AgentError(
            f"Exceeded max iterations ({self.max_iterations}). Transcript:{transcript}"
        )
