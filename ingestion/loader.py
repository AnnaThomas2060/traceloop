import json
import logging
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class Document(BaseModel):
    doc_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    last_updated: str = Field(..., min_length=1)

    @field_validator("doc_id", "title", "content", "source_url", "license", "last_updated", mode="before")
    @classmethod
    def strip_and_validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Field must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be empty or whitespace only")
        return stripped


# Type alias matching Part 1 specification
DocumentRecord = Document


class DocumentLoader:
    """Loads and validates technical documents from a JSONL file."""

    def load(self, path: str | Path) -> list[DocumentRecord]:
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {target_path}")

        documents: list[DocumentRecord] = []
        with open(target_path, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                normalized = unicodedata.normalize("NFKC", raw_line).strip()
                if not normalized:
                    continue
                try:
                    payload = json.loads(normalized)
                    # Verify mandatory keys are present
                    if not all(k in payload for k in ("doc_id", "title", "content")):
                        logger.warning("Skipping line %d missing mandatory keys", line_number)
                        continue
                    doc = Document(**payload)
                    documents.append(doc)
                except Exception as err:
                    logger.warning("Skipping malformed record at line %d: %s", line_number, err)
                    continue

        return documents

    def load_jsonl(self, file_path: str | Path) -> list[Document]:
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {target_path}")

        documents: list[Document] = []
        with open(target_path, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Malformed JSON at line {line_number}: {err}") from err

                try:
                    doc = Document(**payload)
                    documents.append(doc)
                except Exception as err:
                    raise ValueError(f"Invalid document schema at line {line_number}: {err}") from err

        return documents
