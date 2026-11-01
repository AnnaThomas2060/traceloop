from app.qdrant_client import SearchResult, VectorStoreClient, VectorStoreConnectionError


def test_qdrant_client_healthcheck():
    client = VectorStoreClient()
    # Should report healthy since chunks_metadata.jsonl exists
    assert client.is_healthy() is True


def test_qdrant_client_search_top_k_and_schema():
    client = VectorStoreClient()
    dummy_vector = [0.1] * 384
    norm = sum(x * x for x in dummy_vector) ** 0.5
    dummy_vector = [x / norm for x in dummy_vector]

    results = client.search(dummy_vector, top_k=3, score_threshold=0.0)
    assert len(results) <= 3
    for res in results:
        assert isinstance(res, SearchResult)
        assert 0.0 <= res.score <= 1.0
        assert res.chunk_id
        assert res.text
        assert res.doc_id
        assert res.title


def test_score_threshold_filtering():
    client = VectorStoreClient()
    dummy_vector = [0.1] * 384
    norm = sum(x * x for x in dummy_vector) ** 0.5
    dummy_vector = [x / norm for x in dummy_vector]

    # A very high threshold (e.g. 0.999) should filter out most or all hits
    results_strict = client.search(dummy_vector, top_k=10, score_threshold=0.999)
    assert all(r.score >= 0.999 for r in results_strict)


def test_mock_client_injection():
    class MockHit:
        def __init__(self, hid, score, payload):
            self.id = hid
            self.score = score
            self.payload = payload

    class MockQdrant:
        def get_collections(self):
            return []

        def search(self, collection_name, query_vector, limit, score_threshold):
            return [
                MockHit("chunk_99", 0.88, {
                    "chunk_id": "doc_001_chunk_0001",
                    "content": "Indexed content",
                    "doc_id": "doc_001",
                    "title": "Title",
                    "source_url": "https://example.com",
                })
            ]

    client = VectorStoreClient(client=MockQdrant())
    assert client.is_healthy() is True
    res = client.search([0.0] * 384, top_k=1, score_threshold=0.5)
    assert len(res) == 1
    assert res[0].chunk_id == "doc_001_chunk_0001"
    assert res[0].score == 0.88


def test_vector_store_connection_error():
    err = VectorStoreConnectionError("Connection timeout while contacting Qdrant cluster")
    assert isinstance(err, Exception)
    assert "Connection timeout" in str(err)

