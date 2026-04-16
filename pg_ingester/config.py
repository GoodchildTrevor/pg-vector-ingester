from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = Field(
        description="Async DSN: postgresql+asyncpg://user:pass@host/db"
    )
    dense_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="fastembed dense model for message embeddings",
    )
    batch_size: int = Field(
        default=32,
        description="Number of messages to embed per fastembed batch",
    )
    insert_batch_size: int = Field(
        default=64,
        description="Number of UPDATE statements per DB transaction",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().dense_model_name)
