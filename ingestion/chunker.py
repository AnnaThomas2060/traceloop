
from pydantic import BaseModel, Field

from ingestion.loader import Document


class Chunk(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    title: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)

    @property
    def text(self) -> str:
        return self.content


# Type alias matching Part 1 specification
ChunkRecord = Chunk


class DocumentChunker:
    """Splits documents into overlapping chunks using recursive character splitting."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 60,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        raw_text = doc.content
        if len(raw_text) <= self.chunk_size:
            return [
                Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_0000",
                    doc_id=doc.doc_id,
                    chunk_index=0,
                    title=doc.title,
                    source_url=doc.source_url,
                    content=raw_text,
                    char_start=0,
                    char_end=len(raw_text),
                )
            ]

        spans = self._split_text_with_spans(raw_text)
        chunks: list[Chunk] = []

        for idx, (content, start, end) in enumerate(spans):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx:04d}",
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    title=doc.title,
                    source_url=doc.source_url,
                    content=content,
                    char_start=start,
                    char_end=end,
                )
            )

        return chunks

    def split(self, doc: Document) -> list[Chunk]:
        return self.chunk_document(doc)


    def _split_text_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = []
        n = len(text)
        start = 0

        while start < n:
            target_end = min(start + self.chunk_size, n)
            if target_end == n:
                chunk_slice = text[start:n]
                result.append((chunk_slice, start, n))
                break

            best_cut = -1
            for sep in self.separators:
                if not sep:
                    continue
                window = text[start:target_end]
                last_pos = window.rfind(sep)
                if last_pos != -1:
                    cut_candidate = start + last_pos + len(sep)
                    if cut_candidate > start + self.chunk_overlap:
                        best_cut = cut_candidate
                        break

            if best_cut == -1:
                best_cut = target_end

            chunk_slice = text[start:best_cut]
            result.append((chunk_slice, start, best_cut))

            next_start = max(best_cut - self.chunk_overlap, start + 1)
            if next_start >= n:
                break
            start = next_start

        return result
