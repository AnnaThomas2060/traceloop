import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.main import app, ask, get_metrics, health
from app.models import AskRequest, AskResponse, HealthResponse, MetricsResponse


def test_health_handler():
    response = health()
    assert isinstance(response, HealthResponse)
    assert response.status in ["ok", "degraded"]
    assert isinstance(response.qdrant_connected, bool)
    assert response.version == "0.1.0"


def test_ask_conversational_handler():
    req = AskRequest(query="Hello, how are you?")
    res = ask(req)
    assert isinstance(res, AskResponse)
    assert res.tool_used == "fallback_direct_tool"
    assert "no retrieval" in res.answer.lower()
    assert res.latency_ms >= 0.0
    assert res.trace_id
    assert res.retrieved_chunk_ids == []


def test_ask_technical_handler():
    req = AskRequest(query="Explain agent memory consolidation in autonomous systems")
    res = ask(req)
    assert isinstance(res, AskResponse)
    assert res.tool_used == "knowledge_retriever_tool"
    assert res.latency_ms >= 0.0
    assert res.trace_id


def test_ask_validation_empty_query():
    with pytest.raises(ValidationError):
        AskRequest(query="")

    with pytest.raises(ValidationError):
        AskRequest(query="   \n\t  ")


def test_metrics_handler():
    response = get_metrics()
    assert isinstance(response, MetricsResponse)
    assert response.total_requests >= 0
    assert "knowledge_retriever_tool" in response.tool_usage
    assert "fallback_direct_tool" in response.tool_usage


def test_static_html_served():
    static_file = Path("app/static/index.html")
    assert static_file.exists()
    content = static_file.read_text(encoding="utf-8")
    assert "TraceLoop" in content


@pytest.mark.skipif(
    sys.version_info >= (3, 14) and sys.platform == "win32",
    reason="Starlette TestClient with anyio event loop thread hangs on Windows Python 3.14 alpha",
)
def test_testclient_http_integration():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]

    ask_res = client.post("/ask", json={"query": "Hello"})
    assert ask_res.status_code == 200
    assert ask_res.json()["tool_used"] == "fallback_direct_tool"

    err_res = client.post("/ask", json={"query": ""})
    assert err_res.status_code == 422
