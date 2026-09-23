"""HTTP API contract tests via FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from helix.api import create_app
from helix.core.config import Settings
from helix.core.registry import Container


@pytest.fixture
def client():
    c = Container(Settings(_env_file=None, provider_mode="mock"))
    app = create_app(container=c)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "calculator" in body["tools"]


def test_chat(client):
    r = client.post(
        "/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert r.json()["reply"]


def test_embed(client):
    r = client.post("/v1/embed", json={"texts": ["hello", "world"]})
    assert r.status_code == 200
    assert len(r.json()["vectors"]) == 2


def test_rag_flow(client):
    r = client.post(
        "/v1/rag/ingest",
        json={"documents": [{"id": "d1", "text": "Helix supports RAG."}]},
    )
    assert r.status_code == 200 and r.json()["chunks"] >= 1

    r2 = client.post("/v1/rag/query", json={"query": "What does Helix support?"})
    assert r2.status_code == 200 and r2.json()["results"]

    r3 = client.post("/v1/rag/answer", json={"query": "What does Helix support?"})
    assert r3.status_code == 200


def test_agent_run(client):
    r = client.post("/v1/agents/run", json={"task": "do something"})
    assert r.status_code == 200
    assert r.json()["result"]
