import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import get_settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None
    UnexpectedResponse = Exception


class VectorStoreConnectionError(Exception):
    """Raised when communication with the Qdrant vector store fails."""
    pass


class SearchResult(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    text: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=1)


def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


class VectorStoreClient:
    """Manages Qdrant vector search operations with score thresholding and local fallback."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        client: Any | None = None,
        processed_data_path: str | None = None,
    ):
        settings = get_settings()
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.processed_data_path = processed_data_path or "data/processed/chunks_metadata.jsonl"
        self._local_chunks: list[dict[str, Any]] | None = None

        if client is not None:
            self.client = client
        elif HAS_QDRANT and QdrantClient is not None:
            try:
                self.client = QdrantClient(host=self.host, port=self.port, timeout=2.0)
                self.client.get_collections()
            except Exception:
                self.client = None
        else:
            self.client = None

    def is_healthy(self) -> bool:
        if self.client is not None:
            try:
                self.client.get_collections()
                return True
            except Exception:
                return False
        # If client is not connected to a remote Qdrant, check if local chunks exist
        return Path(self.processed_data_path).exists()

    def _load_local_chunks(self) -> list[dict[str, Any]]:
        if self._local_chunks is not None:
            return self._local_chunks

        path = Path(self.processed_data_path)
        if not path.exists():
            return []

        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        self._local_chunks = chunks
        return self._local_chunks

    def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float = 0.65,
    ) -> list[SearchResult]:
        if self.client is not None:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                results: list[SearchResult] = []
                for hit in hits:
                    payload = hit.payload or {}
                    score_val = max(0.0, min(1.0, float(hit.score)))
                    results.append(
                        SearchResult(
                            chunk_id=str(payload.get("chunk_id", hit.id)),
                            score=score_val,
                            text=str(payload.get("content", "")),
                            doc_id=str(payload.get("doc_id", "")),
                            title=str(payload.get("title", "")),
                            source_url=str(payload.get("source_url", "")),
                        )
                    )
                return results
            except Exception:
                pass

        # Fallback local calculation with cached vectors
        from ingestion.embedder import VectorEmbedder

        chunks = self._load_local_chunks()
        if not chunks:
            return []

        if not hasattr(self, "_cached_chunk_vectors") or self._cached_chunk_vectors is None:
            embedder = VectorEmbedder()
            chunk_texts = [f"{c.get('title', '')} {c.get('content', '')}" for c in chunks]
            self._cached_chunk_vectors = embedder.embed_texts(chunk_texts)

        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk_data, c_vec in zip(chunks, self._cached_chunk_vectors):
            sim = dot_product(query_vector, c_vec)
            # Scale cosine similarity for dense feature hashing
            norm_sim = max(0.0, min(1.0, 0.5 + sim * 0.90))
            if norm_sim >= score_threshold:
                scored.append((norm_sim, chunk_data))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored[:top_k]

        return [
            SearchResult(
                chunk_id=data["chunk_id"],
                score=score,
                text=data["content"],
                doc_id=data["doc_id"],
                title=data["title"],
                source_url=data["source_url"],
            )
            for score, data in top_hits
        ]
