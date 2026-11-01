import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False
    BaseSettings = None
    SettingsConfigDict = None


def load_env_file(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not dotenv_path.is_file():
        return values
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return values
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            values[key] = val
    return values


if HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        APP_ENV: str = "development"
        APP_PORT: int = Field(default=8000, ge=1, le=65535)
        OPENAI_API_KEY: str = "mock-key"
        OPENAI_MODEL_NAME: str = "gpt-4o-mini"
        QDRANT_HOST: str = "localhost"
        QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
        QDRANT_COLLECTION: str = "traceloop_documents"
        QDRANT_SCORE_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
        QDRANT_TOP_K: int = Field(default=4, ge=1)
        LANGFUSE_PUBLIC_KEY: str = "pk-lf-local"
        LANGFUSE_SECRET_KEY: str = "sk-lf-local"
        LANGFUSE_HOST: str = "http://localhost:3000"
        MLFLOW_TRACKING_URI: str = "http://localhost:5000"
        MLFLOW_EXPERIMENT_NAME: str = "traceloop-evals"
        REGRESSION_DELTA_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)

        @field_validator("LANGFUSE_HOST", "MLFLOW_TRACKING_URI", mode="before")
        @classmethod
        def strip_trailing_slash(cls, value: Any) -> Any:
            if isinstance(value, str):
                return value.rstrip("/")
            return value

else:
    class Settings(BaseModel):
        APP_ENV: str = "development"
        APP_PORT: int = Field(default=8000, ge=1, le=65535)
        OPENAI_API_KEY: str = "mock-key"
        OPENAI_MODEL_NAME: str = "gpt-4o-mini"
        QDRANT_HOST: str = "localhost"
        QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
        QDRANT_COLLECTION: str = "traceloop_documents"
        QDRANT_SCORE_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
        QDRANT_TOP_K: int = Field(default=4, ge=1)
        LANGFUSE_PUBLIC_KEY: str = "pk-lf-local"
        LANGFUSE_SECRET_KEY: str = "sk-lf-local"
        LANGFUSE_HOST: str = "http://localhost:3000"
        MLFLOW_TRACKING_URI: str = "http://localhost:5000"
        MLFLOW_EXPERIMENT_NAME: str = "traceloop-evals"
        REGRESSION_DELTA_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)

        @field_validator("LANGFUSE_HOST", "MLFLOW_TRACKING_URI", mode="before")
        @classmethod
        def strip_trailing_slash(cls, value: Any) -> Any:
            if isinstance(value, str):
                return value.rstrip("/")
            return value

        def __init__(self, **values: Any):
            env_file_data = load_env_file(Path(".env"))
            combined: dict[str, Any] = {}
            for field_name in self.__class__.model_fields.keys():
                if field_name in values:
                    combined[field_name] = values[field_name]
                elif field_name in os.environ:
                    combined[field_name] = os.environ[field_name]
                elif field_name in env_file_data:
                    combined[field_name] = env_file_data[field_name]
            super().__init__(**combined)


def get_settings() -> Settings:
    return Settings()
