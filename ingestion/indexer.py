import hashlib
import uuid
from typing import Any

from pydantic import BaseModel

from ingestion.chunker import Chunk

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None

FIXED_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class IndexingStats(BaseModel):
    total_documents: int
    total_chunks: int
    collection_point_count: int


class LocalMemoryStore:
    """In-memory mock store preserving Qdrant semantics when running offline."""

    def __init__(self):
        self.collections: dict[str, dict[str, dict[str, Any]]] = {}

    def ensure_collection(self, collection_name: str, dim: int):
        if collection_name not in self.collections:
            self.collections[collection_name] = {}

    def upsert(self, collection_name: str, points: list[dict[str, Any]]):
        self.ensure_collection(collection_name, 384)
        col = self.collections[collection_name]
        for pt in points:
            col[str(pt["id"])] = {
                "id": str(pt["id"]),
                "vector": pt["vector"],
                "payload": pt["payload"],
            }

    def count(self, collection_name: str) -> int:
        if collection_name not in self.collections:
            return 0
        return len(self.collections[collection_name])

    def get_points(self, collection_name: str) -> list[dict[str, Any]]:
        if collection_name not in self.collections:
            return []
        return list(self.collections[collection_name].values())


class QdrantIndexer:
    """Indexes document chunks into Qdrant using deterministic IDs for idempotency."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "traceloop_documents",
        embedding_dim: int = 384,
        client: Any | None = None,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._local_store: LocalMemoryStore | None = None

        if client is not None:
            self.client = client
        elif HAS_QDRANT and QdrantClient is not None:
            try:
                self.client = QdrantClient(host=self.host, port=self.port, timeout=3.0)
                self.client.get_collections()
            except Exception:
                self._local_store = LocalMemoryStore()
                self.client = None
        else:
            self._local_store = LocalMemoryStore()
            self.client = None

    def ensure_collection(self) -> None:
        if self.client is not None:
            try:
                collections = self.client.get_collections().collections
                names = [c.name for c in collections]
                if self.collection_name not in names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
                    )
                return
            except Exception:
                self._local_store = LocalMemoryStore()
                self.client = None

        if self._local_store is None:
            self._local_store = LocalMemoryStore()
        self._local_store.ensure_collection(self.collection_name, self.embedding_dim)

    @staticmethod
    def generate_point_id(doc_id: str, content: str) -> str:
        content_hash = hashlib.sha256(f"{doc_id}:{content}".encode()).hexdigest()
        return str(uuid.uuid5(FIXED_NAMESPACE, content_hash))

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> IndexingStats:
        if len(chunks) != len(vectors):
            raise ValueError("Chunks count and vectors count must match exactly")

        self.ensure_collection()
        distinct_doc_ids = {c.doc_id for c in chunks}

        if self.client is not None:
            points = []
            for chunk, vec in zip(chunks, vectors):
                point_id = self.generate_point_id(chunk.doc_id, chunk.content)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                }
                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)
            count_info = self.client.count(collection_name=self.collection_name)
            total_points = count_info.count
        else:
            local_points = []
            for chunk, vec in zip(chunks, vectors):
                point_id = self.generate_point_id(chunk.doc_id, chunk.content)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source_url": chunk.source_url,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                }
                local_points.append({"id": point_id, "vector": vec, "payload": payload})
            self._local_store.upsert(self.collection_name, local_points)
            total_points = self._local_store.count(self.collection_name)

        return IndexingStats(
            total_documents=len(distinct_doc_ids),
            total_chunks=len(chunks),
            collection_point_count=total_points,
        )

    def upsert_batch(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        stats = self.upsert_chunks(chunks, vectors)
        return stats.collection_point_count


    def get_point_count(self) -> int:
        if self.client is not None:
            return self.client.count(collection_name=self.collection_name).count
        if self._local_store is not None:
            return self._local_store.count(self.collection_name)
        return 0

    def get_local_points(self) -> list[dict[str, Any]]:
        if self._local_store is not None:
            return self._local_store.get_points(self.collection_name)
        return []
