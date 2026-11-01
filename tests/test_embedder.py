import math

from ingestion.embedder import VectorEmbedder


def test_embed_dimension_and_l2_normalization():
    embedder = VectorEmbedder()
    sample_texts = [
        "Agent memory consolidation in distributed systems.",
        "RAGAS evaluation metrics for context recall and faithfulness.",
        "Qdrant HNSW indexing and payload filtering.",
    ]
    vectors = embedder.embed_texts(sample_texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 384
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(l2_norm, 1.0, rel_tol=1e-5)


def test_empty_input_handling():
    embedder = VectorEmbedder()
    vectors = embedder.embed_texts([])
    assert vectors == []


def test_deterministic_vectors():
    embedder = VectorEmbedder()
    text = "Deterministic vector generation check."
    vec1 = embedder.embed_texts([text])[0]
    vec2 = embedder.embed_texts([text])[0]

    assert vec1 == vec2


def test_batch_splitting():
    embedder = VectorEmbedder(batch_size=2)
    texts = [f"Text chunk number {i}" for i in range(5)]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 5
    for vec in vectors:
        assert len(vec) == 384
        l2_norm = math.sqrt(sum(x * x for x in vec))
        assert math.isclose(l2_norm, 1.0, rel_tol=1e-5)
