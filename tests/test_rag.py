"""Tests for the RAG pipeline (split -> embed -> store -> retrieve -> answer)."""
import asyncio

from helix.core.config import Settings
from helix.core.registry import Container
from helix.core.types import Document


def _mock_container():
    return Container(Settings(_env_file=None, provider_mode="mock"))


def test_ingest_and_retrieve():
    async def run():
        c = _mock_container()
        docs = [
            Document(
                id="d1",
                text="Helix is a modular AI engine for RAG and autonomous agents.",
            )
        ]
        n = await c.rag.ingest(docs)
        assert n >= 1
        scored = await c.rag.retrieve("What is Helix?")
        assert scored
        assert "Helix" in scored[0].chunk.text
        answer = await c.rag.answer("What is Helix?", c.llm)
        assert isinstance(answer, str) and answer

    asyncio.run(run())
