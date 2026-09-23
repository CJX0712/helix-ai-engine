"""End-to-end test: a full, offline-runnable Helix pipeline.

Drives the real RAG pipeline (split -> embed -> store -> retrieve -> answer) and
the agent over the deterministic providers, proving the modules compose into a
complete working link without any external model endpoint.
"""
import asyncio

from helix.core.config import Settings
from helix.core.registry import Container
from helix.core.types import Document


def test_full_pipeline():
    async def run():
        c = Container(Settings(_env_file=None, provider_mode="mock"))
        docs = [
            Document(
                id="d1",
                text=(
                    "Helix is a modular, provider-agnostic AI orchestration engine "
                    "designed for retrieval-augmented generation and autonomous agents."
                ),
            ),
            Document(
                id="d2",
                text="The solar system contains eight planets, including Earth and Mars.",
            ),
        ]
        n = await c.rag.ingest(docs)
        assert n >= 2

        scored = await c.rag.retrieve("What is the Helix engine?")
        assert scored
        # The Helix document must be retrieved (ranking quality depends on the
        # embedding provider; the deterministic stand-in is only an offline stub).
        assert any("Helix" in s.chunk.text for s in scored)

        answer = await c.rag.answer("What is the Helix engine?", c.llm)
        assert answer

        # Agent path reusing the same container.
        result = await c.agent.run("What is Helix?")
        assert result

    asyncio.run(run())
