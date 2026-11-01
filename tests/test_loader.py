import json

import pytest

from ingestion.loader import Document, DocumentLoader, DocumentRecord


def test_load_valid_jsonl(tmp_path):
    sample_file = tmp_path / "valid.jsonl"
    record = {
        "doc_id": "doc_001",
        "title": "Test Title",
        "source_url": "https://example.com/test",
        "license": "Apache-2.0",
        "content": "This is valid document content for testing purposes.",
        "last_updated": "2026-03-01",
    }
    sample_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_jsonl(sample_file)
    assert len(docs) == 1
    assert docs[0].doc_id == "doc_001"
    assert docs[0].title == "Test Title"
    assert docs[0].content == "This is valid document content for testing purposes."


def test_load_sample_documents_file():
    loader = DocumentLoader()
    docs = loader.load_jsonl("data/raw/sample_documents.jsonl")
    assert len(docs) == 20
    assert all(isinstance(d, Document) for d in docs)
    assert all(len(d.content.split()) >= 400 for d in docs)


def test_missing_file_raises_not_found():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_jsonl("non_existent_file.jsonl")


def test_reject_whitespace_only_content(tmp_path):
    bad_file = tmp_path / "bad_content.jsonl"
    record = {
        "doc_id": "doc_002",
        "title": "Title",
        "source_url": "https://example.com",
        "license": "MIT",
        "content": "    \n\t  ",
        "last_updated": "2026-03-01",
    }
    bad_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load_jsonl(bad_file)


def test_reject_malformed_json_line(tmp_path):
    bad_file = tmp_path / "broken_json.jsonl"
    bad_file.write_text("{doc_id: invalid_json\n", encoding="utf-8")

    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load_jsonl(bad_file)


def test_whitespace_trimming(tmp_path):
    sample_file = tmp_path / "trim.jsonl"
    record = {
        "doc_id": "  doc_003  ",
        "title": "  Trimmed Title  ",
        "source_url": " https://example.com/trimmed ",
        "license": " Apache-2.0 ",
        "content": "  Proper trimmed content.  ",
        "last_updated": " 2026-03-01 ",
    }
    sample_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load_jsonl(sample_file)
    assert docs[0].doc_id == "doc_003"
    assert docs[0].title == "Trimmed Title"
    assert docs[0].content == "Proper trimmed content."


def test_load_method_skips_malformed_and_preserves_records(tmp_path):
    mixed_file = tmp_path / "mixed.jsonl"
    lines = [
        json.dumps({
            "doc_id": "doc_valid_1",
            "title": "Valid One",
            "source_url": "https://example.com/1",
            "license": "Apache-2.0",
            "content": "Valid content block one.",
            "last_updated": "2026-03-01",
        }),
        "{ malformed json line without closing bracket",
        json.dumps({
            "doc_id": "doc_missing_content",
            "title": "Missing Content",
            "source_url": "https://example.com/2",
            "license": "Apache-2.0",
            "last_updated": "2026-03-01",
        }),
        json.dumps({
            "doc_id": "doc_valid_2",
            "title": "Valid Two",
            "source_url": "https://example.com/3",
            "license": "CC-BY-4.0",
            "content": "Valid content block two.",
            "last_updated": "2026-03-02",
        }),
    ]
    mixed_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loader = DocumentLoader()
    docs = loader.load(mixed_file)
    assert len(docs) == 2
    assert all(isinstance(d, DocumentRecord) for d in docs)
    assert docs[0].doc_id == "doc_valid_1"
    assert docs[1].doc_id == "doc_valid_2"

