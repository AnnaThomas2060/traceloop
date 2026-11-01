import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent import TraceLoopAgent
from app.config import get_settings
from app.models import AskRequest, AskResponse, HealthResponse, MetricsResponse
from app.qdrant_client import VectorStoreClient

settings = get_settings()

app = FastAPI(
    title="TraceLoop Agentic RAG Service",
    description="Local-first agentic RAG service with trace-to-eval regression loop",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime telemetry counters
METRICS = {
    "total_requests": 0,
    "tool_usage": {
        "knowledge_retriever_tool": 0,
        "fallback_direct_tool": 0,
    },
}

agent = TraceLoopAgent(settings=settings)
vector_client = VectorStoreClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    collection_name=settings.QDRANT_COLLECTION,
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    start_time = time.perf_counter()

    try:
        execution = agent.run(query=request.query, session_id=request.session_id)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {err}",
        ) from err

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    METRICS["total_requests"] += 1
    tool_key = execution.tool_used
    METRICS["tool_usage"][tool_key] = METRICS["tool_usage"].get(tool_key, 0) + 1

    return AskResponse(
        answer=execution.answer,
        tool_used=execution.tool_used,
        retrieved_chunk_ids=execution.retrieved_chunk_ids,
        latency_ms=round(elapsed_ms, 2),
        trace_id=execution.trace_id,
        trace_url=execution.trace_url,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_ok = vector_client.is_healthy()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant_connected=qdrant_ok,
        version="0.1.0",
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    return MetricsResponse(
        total_requests=METRICS["total_requests"],
        tool_usage=METRICS["tool_usage"],
    )


# Mount static assets directory
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
index_html = static_dir / "index.html"
if not index_html.exists():
    index_html.write_text("<!DOCTYPE html><html><head><title>TraceLoop</title></head><body><h1>TraceLoop Diagnostic UI</h1></body></html>", encoding="utf-8")

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
