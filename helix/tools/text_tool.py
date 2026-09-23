"""Simple text-transform tool (uppercase / word-count)."""
from __future__ import annotations


class TextTool:
    name = "text"
    description = (
        "Process text. Input format: 'upper: <text>' or 'words: <text>'."
    )

    async def run(self, input: str) -> str:
        text = input.strip()
        if text.lower().startswith("upper:"):
            return text[len("upper:") :].strip().upper()
        if text.lower().startswith("words:"):
            return str(len(text[len("words:") :].strip().split()))
        return text
