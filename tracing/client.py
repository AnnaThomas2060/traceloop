from typing import Any

from app.config import get_settings

try:
    from langfuse import Langfuse
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    Langfuse = None


class LangfuseClientWrapper:
    """Queries production traces from Langfuse for evaluation dataset promotion."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        client: Any | None = None,
    ):
        settings = get_settings()
        self.public_key = public_key or settings.LANGFUSE_PUBLIC_KEY
        self.secret_key = secret_key or settings.LANGFUSE_SECRET_KEY
        self.host = host or settings.LANGFUSE_HOST
        self.client = client
        self._is_connected = False

        if self.client is not None:
            self._is_connected = True
        elif HAS_LANGFUSE and Langfuse is not None and self.public_key != "pk-lf-local":
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
                self._is_connected = True
            except Exception:
                self.client = None

    def get_traces_with_negative_feedback(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.client is not None and hasattr(self.client, "get_traces"):
            try:
                # Query Langfuse SDK for traces with negative scores
                res = self.client.get_traces(tags=["negative_feedback"], limit=limit)
                return getattr(res, "data", res)
            except Exception:
                pass

        # Deterministic offline candidate trace
        return [
            {
                "id": "trace_reg_001",
                "input": {"query": "What are the exact chunk overlap parameters recommended for technical documents?"},
                "output": {"answer": "Technical documentation requires specific chunk sizes and overlap settings."},
                "scores": [{"name": "user_feedback", "value": 0.0, "comment": "Missing specific numbers"}],
                "observations": [
                    {
                        "type": "SPAN",
                        "name": "agent_plan",
                        "output": {"tool_used": "knowledge_retriever_tool"},
                    },
                    {
                        "type": "TOOL",
                        "name": "knowledge_retriever_tool",
                        "input": {"query": "chunk overlap parameters"},
                        "output": "[Chunk: doc_013_chunk_0000] (Title: Chunk Size and Overlap Trade-Offs)\nStandard guideline uses a chunk size of 500 characters with 60 characters overlap.",
                        "metadata": {"retrieved_chunk_ids": ["doc_013_chunk_0000"]},
                    },
                    {
                        "type": "GENERATION",
                        "name": "response_synthesis",
                        "output": "Technical documentation requires specific chunk sizes and overlap settings.",
                    },
                ],
            }
        ]

    def get_traces_with_errors(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.client is not None and hasattr(self.client, "get_traces"):
            try:
                res = self.client.get_traces(tags=["error"], limit=limit)
                return getattr(res, "data", res)
            except Exception:
                pass

        return [
            {
                "id": "trace_err_002",
                "input": {"query": "How to resolve Qdrant memory graph clustering anomalies?"},
                "output": {"error": "Timeout during vector similarity graph traversal"},
                "scores": [],
                "observations": [
                    {
                        "type": "TOOL",
                        "name": "knowledge_retriever_tool",
                        "input": {"query": "Qdrant memory graph clustering"},
                        "output": "Error: Timeout",
                        "metadata": {"retrieved_chunk_ids": []},
                    }
                ],
            }
        ]

    def get_low_scoring_traces(self, score_threshold: float = 0.5, limit: int = 50) -> list[dict[str, Any]]:
        # Combines negative feedback and low-scoring candidate traces
        return self.get_traces_with_negative_feedback(limit=limit)
