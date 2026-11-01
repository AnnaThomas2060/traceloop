from typing import Any

from pydantic import BaseModel, Field

from app.qdrant_client import VectorStoreClient
from ingestion.embedder import VectorEmbedder

try:
    from langchain_core.tools import StructuredTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    StructuredTool = None


class KnowledgeRetrieverInput(BaseModel):
    query: str = Field(..., min_length=1, description="Technical search query for documentation.")


class FallbackDirectInput(BaseModel):
    query: str = Field(..., min_length=1, description="Conversational greeting or general query.")


class ToolExecutionResult(BaseModel):
    output: str
    retrieved_chunk_ids: list[str]
    retrieval_used: bool


class KnowledgeRetrieverTool:
    """Tool that embeds queries, searches Qdrant, and formats context chunks."""

    name: str = "knowledge_retriever_tool"
    description: str = (
        "Use this tool for technical queries regarding agent memory consolidation, "
        "RAG evaluation metrics, vector search optimization, and observability."
    )
    args_schema: type[BaseModel] = KnowledgeRetrieverInput

    def __init__(
        self,
        embedder: VectorEmbedder | None = None,
        vector_client: VectorStoreClient | None = None,
        top_k: int = 4,
        score_threshold: float = 0.65,
    ):
        self.embedder = embedder or VectorEmbedder()
        self.vector_client = vector_client or VectorStoreClient()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.last_metadata: dict[str, Any] = {"retrieved_chunk_ids": [], "retrieval_used": True}

    def run(self, query: str) -> str:
        validated = KnowledgeRetrieverInput(query=query)
        query_vectors = self.embedder.embed_texts([validated.query])
        if not query_vectors:
            self.last_metadata = {"retrieved_chunk_ids": [], "retrieval_used": True}
            return "No context retrieved."

        results = self.vector_client.search(
            query_vector=query_vectors[0],
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        chunk_ids = [r.chunk_id for r in results]
        self.last_metadata = {"retrieved_chunk_ids": chunk_ids, "retrieval_used": True}

        if not results:
            return "No relevant context found in documentation."

        formatted_chunks: list[str] = []
        for r in results:
            formatted_chunks.append(f"[Chunk: {r.chunk_id}] (Title: {r.title})\n{r.text}")

        return "\n\n".join(formatted_chunks)

    def __call__(self, query: str) -> str:
        return self.run(query)


class FallbackDirectTool:
    """Tool for conversational messages that bypasses vector retrieval."""

    name: str = "fallback_direct_tool"
    description: str = (
        "Use this tool for conversational greetings, basic pleasantries, or general queries "
        "outside indexed technical topics."
    )
    args_schema: type[BaseModel] = FallbackDirectInput

    def __init__(self):
        self.last_metadata: dict[str, Any] = {"retrieved_chunk_ids": [], "retrieval_used": False}

    def run(self, query: str) -> str:
        FallbackDirectInput(query=query)
        self.last_metadata = {"retrieved_chunk_ids": [], "retrieval_used": False}
        return (
            "Direct conversational response: no retrieval was performed for this query. "
            "I can assist you with technical documentation on agent memory, RAG metrics, "
            "vector databases, and observability."
        )

    def __call__(self, query: str) -> str:
        return self.run(query)


def get_langchain_tools(
    retriever_tool: KnowledgeRetrieverTool | None = None,
    fallback_tool: FallbackDirectTool | None = None,
) -> list[Any]:
    rt = retriever_tool or KnowledgeRetrieverTool()
    ft = fallback_tool or FallbackDirectTool()

    if HAS_LANGCHAIN and StructuredTool is not None:
        lc_retriever = StructuredTool.from_function(
            func=rt.run,
            name=rt.name,
            description=rt.description,
            args_schema=rt.args_schema,
        )
        lc_fallback = StructuredTool.from_function(
            func=ft.run,
            name=ft.name,
            description=ft.description,
            args_schema=ft.args_schema,
        )
        return [lc_retriever, lc_fallback]

    return [rt, ft]
