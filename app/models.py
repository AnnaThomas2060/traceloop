
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query string")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")

    @field_validator("query", mode="before")
    @classmethod
    def validate_non_empty_query(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Query must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be empty or whitespace only")
        return stripped


class AskResponse(BaseModel):
    answer: str
    tool_used: str
    retrieved_chunk_ids: list[str]
    latency_ms: float
    trace_id: str
    trace_url: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    version: str = "0.1.0"


class MetricsResponse(BaseModel):
    total_requests: int
    tool_usage: dict[str, int]
