import pytest

from ingestion.chunker import ChunkRecord, DocumentChunker
from ingestion.loader import Document


def test_short_document_creates_single_chunk():
    doc = Document(
        doc_id="doc_short",
        title="Short Doc",
        source_url="https://example.com/short",
        license="Apache-2.0",
        content="This is a short document well below five hundred characters.",
        last_updated="2026-03-01",
    )
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=60)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "doc_short"
    assert chunks[0].title == "Short Doc"
    assert chunks[0].content == doc.content
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(doc.content)


def test_long_document_splits_and_respects_chunk_size():
    doc = Document(
        doc_id="doc_long",
        title="Long Doc",
        source_url="https://example.com/long",
        license="Apache-2.0",
        content="Sentence one. " * 80,
        last_updated="2026-03-01",
    )
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 300
        assert chunk.doc_id == "doc_long"
        assert chunk.title == "Long Doc"
        assert chunk.source_url == "https://example.com/long"


def test_overlap_preservation():
    text = (
        "Paragraph one discusses agent memory consolidation mechanisms in autonomous systems.\n\n"
        "Paragraph two analyzes vector indexing in Qdrant with HNSW graph parameters.\n\n"
        "Paragraph three covers evaluation metrics including context recall and faithfulness."
    )
    doc = Document(
        doc_id="doc_overlap",
        title="Overlap Test",
        source_url="https://example.com/overlap",
        license="MIT",
        content=text,
        last_updated="2026-03-01",
    )
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert c2.char_start < c1.char_end


def test_invalid_parameters():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=0)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=150)


def test_split_method_and_chunk_record_text_property():
    doc = Document(
        doc_id="doc_split_test",
        title="Split Method Test",
        source_url="https://example.com/split",
        license="Apache-2.0",
        content="Testing the split method alias and text property attribute.",
        last_updated="2026-03-01",
    )
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=60)
    chunks = chunker.split(doc)

    assert len(chunks) == 1
    assert isinstance(chunks[0], ChunkRecord)
    assert chunks[0].text == doc.content
    assert chunks[0].text == chunks[0].content

