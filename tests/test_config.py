import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from ingestion.config import get_ingestion_config


def test_default_settings():
    settings = Settings()
    assert settings.APP_ENV == "development"
    assert settings.APP_PORT == 8000
    assert settings.OPENAI_API_KEY == "mock-key"
    assert settings.OPENAI_MODEL_NAME == "gpt-4o-mini"
    assert settings.QDRANT_HOST == "localhost"
    assert settings.QDRANT_PORT == 6333
    assert settings.QDRANT_COLLECTION == "traceloop_documents"
    assert settings.QDRANT_SCORE_THRESHOLD == 0.65
    assert settings.QDRANT_TOP_K == 4
    assert settings.LANGFUSE_PUBLIC_KEY == "pk-lf-local"
    assert settings.LANGFUSE_SECRET_KEY == "sk-lf-local"
    assert settings.LANGFUSE_HOST == "http://localhost:3000"
    assert settings.MLFLOW_TRACKING_URI == "http://localhost:5000"
    assert settings.MLFLOW_EXPERIMENT_NAME == "traceloop-evals"
    assert settings.REGRESSION_DELTA_THRESHOLD == 0.05


def test_override_settings_via_env(monkeypatch):
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_SCORE_THRESHOLD", "0.80")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.internal:3000/")

    settings = Settings()
    assert settings.APP_PORT == 9000
    assert settings.QDRANT_HOST == "qdrant.internal"
    assert settings.QDRANT_SCORE_THRESHOLD == 0.80
    assert settings.LANGFUSE_HOST == "http://langfuse.internal:3000"


def test_trailing_slash_normalization():
    settings = Settings(
        LANGFUSE_HOST="http://localhost:3000///",
        MLFLOW_TRACKING_URI="http://localhost:5000/",
    )
    assert settings.LANGFUSE_HOST == "http://localhost:3000"
    assert settings.MLFLOW_TRACKING_URI == "http://localhost:5000"


def test_invalid_port_validation():
    with pytest.raises(ValidationError):
        Settings(APP_PORT=-1)

    with pytest.raises(ValidationError):
        Settings(APP_PORT=70000)


def test_invalid_score_threshold_validation():
    with pytest.raises(ValidationError):
        Settings(QDRANT_SCORE_THRESHOLD=1.5)

    with pytest.raises(ValidationError):
        Settings(QDRANT_SCORE_THRESHOLD=-0.1)


def test_get_settings_helper():
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_ingestion_config_defaults():
    config = get_ingestion_config()
    assert config.RAW_DATA_PATH == "data/raw/sample_documents.jsonl"
    assert config.PROCESSED_DATA_PATH == "data/processed/chunks_metadata.jsonl"
    assert config.CHUNK_SIZE == 500
    assert config.CHUNK_OVERLAP == 60
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert config.EMBEDDING_DIM == 384
