import pytest
from pydantic import ValidationError

from app.qdrant_client import SearchResult
from app.tools import (
    FallbackDirectInput,
    FallbackDirectTool,
    KnowledgeRetrieverInput,
    KnowledgeRetrieverTool,
    get_langchain_tools,
)


def test_fallback_tool_contains_no_retrieval_token():
    tool = FallbackDirectTool()
    output = tool.run("Hello, how can you help me?")

    assert "no retrieval" in output.lower()
    assert tool.last_metadata["retrieval_used"] is False
    assert tool.last_metadata["retrieved_chunk_ids"] == []


def test_knowledge_retriever_formatting():
    class MockVectorClient:
        def search(self, query_vector, top_k, score_threshold):
            return [
                SearchResult(
                    chunk_id="doc_001_chunk_0001",
                    score=0.88,
                    text="First retrieved chunk content.",
                    doc_id="doc_001",
                    title="Doc One Title",
                    source_url="https://example.com/1",
                ),
                SearchResult(
                    chunk_id="doc_002_chunk_0002",
                    score=0.75,
                    text="Second retrieved chunk content.",
                    doc_id="doc_002",
                    title="Doc Two Title",
                    source_url="https://example.com/2",
                ),
            ]

    tool = KnowledgeRetrieverTool(vector_client=MockVectorClient())
    output = tool.run("How does agent memory consolidation work?")

    assert "[Chunk: doc_001_chunk_0001] (Title: Doc One Title)" in output
    assert "First retrieved chunk content." in output
    assert "[Chunk: doc_002_chunk_0002] (Title: Doc Two Title)" in output
    assert tool.last_metadata["retrieval_used"] is True
    assert tool.last_metadata["retrieved_chunk_ids"] == [
        "doc_001_chunk_0001",
        "doc_002_chunk_0002",
    ]


def test_input_validation():
    with pytest.raises(ValidationError):
        KnowledgeRetrieverInput(query="")

    with pytest.raises(ValidationError):
        FallbackDirectInput(query="")


def test_get_langchain_tools_list():
    tools = get_langchain_tools()
    assert len(tools) == 2
    names = [t.name for t in tools]
    assert "knowledge_retriever_tool" in names
    assert "fallback_direct_tool" in names
