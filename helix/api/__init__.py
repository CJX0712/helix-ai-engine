"""FastAPI application exposing the Helix engine over HTTP."""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel

from ..core.registry import Container
from ..core.types import Document, Message, Role


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str


class DocumentIn(BaseModel):
    id: str
    text: str
    metadata: Dict[str, object] = {}


class IngestRequest(BaseModel):
    documents: List[DocumentIn]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class AgentRequest(BaseModel):
    task: str


def _make_lifespan(container: Optional[Container], settings):
    async def lifespan(app: FastAPI):
        if container is not None:
            app.state.container = container
        else:
            from ..core.config import settings as default_settings

            app.state.container = Container(settings or default_settings)
        yield
        try:
            await app.state.container.close()
        except Exception:
            pass

    return lifespan


def create_app(
    container: Optional[Container] = None, settings=None
) -> FastAPI:
    app = FastAPI(
        title="Helix AI Engine",
        version="1.0.0",
        description="Modular, provider-agnostic AI orchestration engine.",
        lifespan=_make_lifespan(container, settings),
    )

    @app.get("/health")
    async def health(request: Request):
        c: Container = request.app.state.container
        return {
            "status": "ok",
            "provider_mode": c.settings.provider_mode,
            "llm": c.llm.name,
            "embeddings": c.embeddings.name,
            "vectorstore": c.vectorstore.name,
            "tools": list(c.tools.keys()),
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: Request, payload: ChatRequest):
        c: Container = request.app.state.container
        messages = [
            Message(role=Role(m["role"]), content=m["content"])
            for m in payload.messages
        ]
        reply = await c.llm.generate(messages)
        return ChatResponse(reply=reply)

    @app.post("/v1/embed")
    async def embed(request: Request, payload: Dict):
        c: Container = request.app.state.container
        texts = payload.get("texts", [])
        vectors = await c.embeddings.embed(texts)
        return {"vectors": vectors, "dimension": c.embeddings.dimension}

    @app.post("/v1/rag/ingest")
    async def rag_ingest(request: Request, payload: IngestRequest):
        c: Container = request.app.state.container
        docs = [
            Document(id=d.id, text=d.text, metadata=d.metadata)
            for d in payload.documents
        ]
        n = await c.rag.ingest(docs)
        return {"chunks": n}

    @app.post("/v1/rag/query")
    async def rag_query(request: Request, payload: QueryRequest):
        c: Container = request.app.state.container
        scored = await c.rag.retrieve(payload.query, payload.top_k)
        return {
            "results": [
                {
                    "document_id": s.chunk.document_id,
                    "index": s.chunk.index,
                    "score": s.score,
                    "text": s.chunk.text,
                }
                for s in scored
            ]
        }

    @app.post("/v1/rag/answer")
    async def rag_answer(request: Request, payload: QueryRequest):
        c: Container = request.app.state.container
        answer = await c.rag.answer(payload.query, c.llm)
        return {"answer": answer}

    @app.post("/v1/agents/run")
    async def agent_run(request: Request, payload: AgentRequest):
        c: Container = request.app.state.container
        result = await c.agent.run(payload.task)
        return {"result": result}

    return app
