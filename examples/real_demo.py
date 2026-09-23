"""Real-model end-to-end demo against a local Ollama instance.

Run with:  python examples/real_demo.py
Requires Ollama serving the OpenAI-compatible API at HELIX_LLM_BASE_URL.
"""
import os

# Configure the engine to talk to the local Ollama OpenAI-compatible endpoint.
os.environ.setdefault("HELIX_PROVIDER_MODE", "auto")
os.environ.setdefault("HELIX_LLM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("HELIX_LLM_API_KEY", "ollama")
os.environ.setdefault("HELIX_LLM_MODEL", "qwen2.5:1.5b-instruct")
os.environ.setdefault("HELIX_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("HELIX_EMBEDDING_API_KEY", "ollama")
os.environ.setdefault("HELIX_EMBEDDING_MODEL", "nomic-embed-text:latest")

from helix.core.config import settings
from helix.core.registry import Container
from helix.core.types import Document, Message, Role

import asyncio


async def main() -> None:
    c = Container(settings)
    print(f"[providers] llm={c.llm.name} embeddings={c.embeddings.name}")

    # 1) Direct chat with the real model.
    chat = await c.llm.generate(
        [Message(role=Role.USER, content="用一句话介绍 Helix 引擎。")]
    )
    print("\n[chat]\n", chat)

    # 2) RAG: ingest -> embed (real) -> retrieve -> answer (real LLM).
    docs = [
        Document(
            id="d1",
            text="Helix 是一套模块化、供应商无关的 AI 编排引擎，支持检索增强生成(RAG)与工具调用智能体。",
        ),
        Document(
            id="d2",
            text="向量检索负责把用户查询映射到最相关的文档片段，是 RAG 的核心环节。",
        ),
    ]
    n = await c.rag.ingest(docs)
    print(f"\n[rag] ingested {n} chunk(s)")
    answer = await c.rag.answer("Helix 是什么？", c.llm)
    print("[rag answer]\n", answer)

    # 3) Agent: real LLM drives the tool-use loop.
    result = await c.agent.run("请计算 (12 + 8) * 3 等于多少？")
    print("\n[agent]\n", result)

    await c.close()


if __name__ == "__main__":
    asyncio.run(main())
