
import pytest

from ingestion.chunker import Chunk
from ingestion.loader import Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        doc_id="doc_fixture_001",
        title="Fixture Document Title",
        source_url="https://docs.traceloop.dev/fixture",
        license="Apache-2.0",
        content="This is sample content used across unit tests for validation and chunking.",
        last_updated="2026-03-01",
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="doc_fixture_001_chunk_0000",
        doc_id="doc_fixture_001",
        chunk_index=0,
        title="Fixture Document Title",
        source_url="https://docs.traceloop.dev/fixture",
        content="This is sample content used across unit tests for validation and chunking.",
        char_start=0,
        char_end=74,
    )


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    vec = [0.05] * 384
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]
