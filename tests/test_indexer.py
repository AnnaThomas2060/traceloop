from typing import Any

import pytest

from ingestion.chunker import Chunk
from ingestion.indexer import LocalMemoryStore, QdrantIndexer


def make_sample_chunk(idx: int, doc_id: str = "doc_001") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_chunk_{idx:04d}",
        doc_id=doc_id,
        chunk_index=idx,
        title="Sample Architecture Guide",
        source_url="https://example.com/guide",
        content=f"Content for chunk number {idx} regarding vector indexing.",
        char_start=idx * 50,
        char_end=(idx + 1) * 50,
    )


def test_deterministic_point_id():
    id1 = QdrantIndexer.generate_point_id("doc_001", "Identical content block")
    id2 = QdrantIndexer.generate_point_id("doc_001", "Identical content block")
    id3 = QdrantIndexer.generate_point_id("doc_002", "Identical content block")

    assert id1 == id2
    assert id1 != id3


def test_indexer_idempotency_local_store():
    store = LocalMemoryStore()
    indexer = QdrantIndexer(collection_name="test_col", client=None)
    indexer._local_store = store

    chunks = [make_sample_chunk(i) for i in range(3)]
    vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

    stats_first = indexer.upsert_chunks(chunks, vectors)
    assert stats_first.total_chunks == 3
    assert stats_first.collection_point_count == 3
    assert indexer.get_point_count() == 3

    # Re-run upsert on the exact same chunks
    stats_second = indexer.upsert_chunks(chunks, vectors)
    assert stats_second.total_chunks == 3
    assert stats_second.collection_point_count == 3
    assert indexer.get_point_count() == 3


def test_payload_schema_keys():
    store = LocalMemoryStore()
    indexer = QdrantIndexer(collection_name="test_payload", client=None)
    indexer._local_store = store

    chunk = make_sample_chunk(0)
    vector = [0.05] * 384
    indexer.upsert_chunks([chunk], [vector])

    points = indexer.get_local_points()
    assert len(points) == 1
    payload = points[0]["payload"]
    expected_keys = {"chunk_id", "doc_id", "title", "source_url", "content", "chunk_index"}
    assert expected_keys.issubset(payload.keys())
    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["doc_id"] == chunk.doc_id
    assert payload["title"] == chunk.title


def test_chunk_vector_count_mismatch_raises_error():
    indexer = QdrantIndexer(collection_name="test_mismatch", client=None)
    chunks = [make_sample_chunk(0)]
    vectors = []

    with pytest.raises(ValueError):
        indexer.upsert_chunks(chunks, vectors)


def test_indexer_idempotency_with_mock_qdrant_client():
    class MockCount:
        def __init__(self, count: int):
            self.count = count

    class MockCollections:
        def __init__(self, collections: list[Any]):
            self.collections = collections

    class MockCollectionInfo:
        def __init__(self, name: str):
            self.name = name

    class MockQdrantClient:
        def __init__(self):
            self.collections: dict[str, dict[str, Any]] = {}

        def get_collections(self):
            return MockCollections([MockCollectionInfo(c) for c in self.collections.keys()])

        def create_collection(self, collection_name, vectors_config):
            self.collections[collection_name] = {}

        def upsert(self, collection_name, points):
            if collection_name not in self.collections:
                self.collections[collection_name] = {}
            for pt in points:
                pt_id = getattr(pt, "id", str(pt.get("id") if isinstance(pt, dict) else pt))
                self.collections[collection_name][str(pt_id)] = pt

        def count(self, collection_name):
            return MockCount(len(self.collections.get(collection_name, {})))

    mock_client = MockQdrantClient()
    indexer = QdrantIndexer(collection_name="qdrant_mock_col", client=mock_client)

    chunks = [make_sample_chunk(i) for i in range(4)]
    vectors = [[0.1 * i] * 384 for i in range(4)]

    # Initial upsert
    stats1 = indexer.upsert_chunks(chunks, vectors)
    assert stats1.total_documents == 1
    assert stats1.total_chunks == 4
    assert stats1.collection_point_count == 4
    assert indexer.get_point_count() == 4

    # Idempotent re-run on identical data
    stats2 = indexer.upsert_chunks(chunks, vectors)
    assert stats2.total_documents == 1
    assert stats2.total_chunks == 4
    assert stats2.collection_point_count == 4
    assert indexer.get_point_count() == 4


def test_indexer_upsert_batch_returns_point_count():
    indexer = QdrantIndexer(collection_name="batch_test_col", client=None)
    chunks = [make_sample_chunk(i) for i in range(3)]
    vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

    point_count = indexer.upsert_batch(chunks, vectors)
    assert point_count == 3

    # Idempotent re-run returns same count
    point_count_repeat = indexer.upsert_batch(chunks, vectors)
    assert point_count_repeat == 3

