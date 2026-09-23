"""Helix command-line interface (argparse, zero extra dependencies)."""
from __future__ import annotations

import argparse
import asyncio
import sys

from ..core.config import settings
from ..core.registry import Container
from ..core.types import Document, Message, Role


async def _chat(container: Container, prompt: str) -> None:
    reply = await container.llm.generate([Message(role=Role.USER, content=prompt)])
    print(reply)


async def _rag_demo(container: Container) -> None:
    docs = [
        Document(
            id="d1",
            text=(
                "Helix is a modular AI orchestration engine. It splits documents, "
                "embeds them, stores vectors, and answers questions via RAG. "
                "It also supports tool-using agents."
            ),
            metadata={"source": "intro"},
        )
    ]
    n = await container.rag.ingest(docs)
    print(f"Ingested {n} chunk(s).")
    scored = await container.rag.retrieve("What is Helix?")
    for s in scored:
        print(f"  score={s.score:.3f} :: {s.chunk.text[:60]}")
    answer = await container.rag.answer("What is Helix?", container.llm)
    print("Answer:", answer)


async def _agent_demo(container: Container, task: str) -> None:
    print(await container.agent.run(task))


async def _serve() -> None:
    import uvicorn

    from ..api import create_app

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main(argv: list = None) -> None:
    parser = argparse.ArgumentParser(prog="helix", description="Helix AI engine CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve").set_defaults(func=lambda c: _serve())
    p_chat = sub.add_parser("chat", help="Send a single prompt to the LLM")
    p_chat.add_argument("prompt")
    sub.add_parser("rag", help="Run a built-in RAG demo")
    p_agent = sub.add_parser("agent", help="Run the agent on a task")
    p_agent.add_argument("task")

    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        sys.exit(1)

    container = Container(settings)
    if args.cmd == "serve":
        asyncio.run(_serve())
    elif args.cmd == "chat":
        asyncio.run(_chat(container, args.prompt))
    elif args.cmd == "rag":
        asyncio.run(_rag_demo(container))
    elif args.cmd == "agent":
        asyncio.run(_agent_demo(container, args.task))


if __name__ == "__main__":
    main()
