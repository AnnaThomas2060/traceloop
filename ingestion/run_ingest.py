import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from ingestion.chunker import DocumentChunker
from ingestion.config import get_ingestion_config
from ingestion.embedder import VectorEmbedder
from ingestion.indexer import QdrantIndexer
from ingestion.loader import DocumentLoader


def run_pipeline(
    raw_path: str,
    processed_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
) -> dict[str, int]:
    loader = DocumentLoader()
    documents = loader.load_jsonl(raw_path)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    # Save processed chunks metadata
    out_path = Path(processed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(chunk.model_dump()) + "\n" for chunk in all_chunks)

    embedder = VectorEmbedder(model_name=embedding_model)
    texts = [c.content for c in all_chunks]
    vectors = embedder.embed_texts(texts)

    indexer = QdrantIndexer(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    stats = indexer.upsert_chunks(all_chunks, vectors)

    print(
        f"Ingestion pipeline completed successfully. Loaded {stats.total_documents} source documents "
        f"and indexed {stats.total_chunks} chunks into Qdrant collection {collection_name}. "
        f"The verified collection point count is {stats.collection_point_count}."
    )

    return {
        "documents": stats.total_documents,
        "chunks": stats.total_chunks,
        "collection_points": stats.collection_point_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Execute TraceLoop document ingestion pipeline.")
    ingest_cfg = get_ingestion_config()
    app_cfg = get_settings()

    parser.add_argument("--raw-data", default=ingest_cfg.RAW_DATA_PATH)
    parser.add_argument("--processed-data", default=ingest_cfg.PROCESSED_DATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=ingest_cfg.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=ingest_cfg.CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=ingest_cfg.EMBEDDING_MODEL)
    parser.add_argument("--qdrant-host", default=app_cfg.QDRANT_HOST)
    parser.add_argument("--qdrant-port", type=int, default=app_cfg.QDRANT_PORT)
    parser.add_argument("--collection-name", default=app_cfg.QDRANT_COLLECTION)

    args = parser.parse_args()

    run_pipeline(
        raw_path=args.raw_data,
        processed_path=args.processed_data,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
