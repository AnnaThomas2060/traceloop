from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    RAW_DATA_PATH: str = "data/raw/sample_documents.jsonl"
    PROCESSED_DATA_PATH: str = "data/processed/chunks_metadata.jsonl"
    CHUNK_SIZE: int = Field(default=500, gt=0)
    CHUNK_OVERLAP: int = Field(default=60, ge=0)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = Field(default=384, gt=0)


def get_ingestion_config() -> IngestionConfig:
    return IngestionConfig()
