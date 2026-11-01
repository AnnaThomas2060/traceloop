import json

from tracing.sync_dataset import compute_query_hash, parse_trace, sync_traces_to_dataset


def test_parse_valid_trace():
    trace = {
        "id": "trace_001",
        "input": {"query": "Explain HNSW indexing in Qdrant"},
        "output": {"answer": "HNSW creates multi-layer graphs."},
        "observations": [
            {
                "type": "TOOL",
                "name": "knowledge_retriever_tool",
                "input": {"query": "HNSW indexing in Qdrant"},
                "output": "[Chunk: doc_005_chunk_0000] HNSW details...",
                "metadata": {"retrieved_chunk_ids": ["doc_005_chunk_0000"]},
            },
            {
                "type": "GENERATION",
                "name": "response_synthesis",
                "output": "HNSW creates multi-layer graphs.",
            },
        ],
    }

    parsed = parse_trace(trace)
    assert parsed is not None
    assert parsed["question"] == "Explain HNSW indexing in Qdrant"
    assert parsed["tool_used"] == "knowledge_retriever_tool"
    assert parsed["retrieved_chunk_ids"] == ["doc_005_chunk_0000"]
    assert "HNSW details" in parsed["retrieved_context"]
    assert parsed["generated_answer"] == "HNSW creates multi-layer graphs."
    assert parsed["query_hash"] == compute_query_hash("Explain HNSW indexing in Qdrant")


def test_parse_malformed_trace_returns_none():
    assert parse_trace({}) is None
    assert parse_trace({"input": {}}) is None
    assert parse_trace({"input": {"query": "   "}}) is None


def test_sync_deduplication_and_file_append(tmp_path):
    out_file = tmp_path / "promoted_traces.jsonl"

    class MockLangfuseClient:
        def get_traces_with_negative_feedback(self):
            return [
                {
                    "id": "trace_mock_1",
                    "input": {"query": "What is scalar quantization?"},
                    "output": {"answer": "Scalar quantization compresses vectors."},
                    "observations": [
                        {
                            "type": "TOOL",
                            "name": "knowledge_retriever_tool",
                            "output": "[Chunk: doc_007_chunk_0000] Quantization details",
                            "metadata": {"retrieved_chunk_ids": ["doc_007_chunk_0000"]},
                        }
                    ],
                }
            ]

        def get_traces_with_errors(self):
            return []

    client = MockLangfuseClient()

    # First sync run: appends 1 record
    res1 = sync_traces_to_dataset(output_path=str(out_file), client=client)
    assert res1["new_appended"] == 1
    assert res1["duplicates_skipped"] == 0

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["question"] == "What is scalar quantization?"

    # Second sync run with same query: skips duplicate
    res2 = sync_traces_to_dataset(output_path=str(out_file), client=client)
    assert res2["new_appended"] == 0
    assert res2["duplicates_skipped"] == 1

    # Total lines in file remains 1
    lines_after = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) == 1
