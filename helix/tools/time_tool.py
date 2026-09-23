"""Current-time tool."""
from __future__ import annotations

from datetime import datetime, timezone


class TimeTool:
    name = "time"
    description = "Return the current UTC date and time."

    async def run(self, input: str) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
