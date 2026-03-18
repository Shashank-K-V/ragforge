"""
config.py — Centralised configuration for RAGForge.

All settings are read from environment variables (with sensible defaults)
so the app works identically in development, Docker, and production without
any code changes — just swap .env files.

Priority (highest → lowest):
  1. Real environment variables  (set by shell / docker-compose / HF Spaces secrets)
  2. Values in a .env file       (loaded automatically by pydantic-settings)
  3. Default values below
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings resolved from environment variables."""

    # ------------------------------------------------------------------ #
    # App metadata                                                         #
    # ------------------------------------------------------------------ #
    APP_NAME: str = "RAGForge"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-ready Document Intelligence REST API powered by RAG."
    )

    # ------------------------------------------------------------------ #
    # API / server                                                         #
    # ------------------------------------------------------------------ #
    HOST: str = "0.0.0.0"
    PORT: int = 7860          # 7860 is the default exposed port on HF Spaces
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # LLM provider                                                         #
    # ------------------------------------------------------------------ #
    # Set LLM_PROVIDER=openai and supply OPENAI_API_KEY to use GPT-3.5/4.
    # Default is "huggingface" — uses the free HF Inference API.
    LLM_PROVIDER: Literal["huggingface", "openai"] = "huggingface"

    # HuggingFace Inference API (free tier — no credit card required)
    HUGGINGFACE_API_KEY: str = ""
    # Model served by HF Inference API.  mistralai/Mistral-7B-Instruct-v0.2
    # is a strong open-weight model available on the free tier.
    HF_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # OpenAI (optional upgrade)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # ------------------------------------------------------------------ #
    # Embeddings                                                           #
    # ------------------------------------------------------------------ #
    # sentence-transformers runs fully locally — no API key needed.
    # all-MiniLM-L6-v2 is small (80 MB), fast, and works great for QA.
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ------------------------------------------------------------------ #
    # Vector store (ChromaDB — local, fully free)                         #
    # ------------------------------------------------------------------ #
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "ragforge_docs"

    # ------------------------------------------------------------------ #
    # Document ingestion                                                   #
    # ------------------------------------------------------------------ #
    # Chunk size in *characters* (≈ 500 tokens for English prose at
    # ~4 chars/token).  Overlap preserves cross-boundary context.
    CHUNK_SIZE: int = 2000        # ~500 tokens
    CHUNK_OVERLAP: int = 200      # ~50 tokens
    MAX_FILE_SIZE_MB: int = 50    # reject uploads larger than this

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #
    RETRIEVAL_TOP_K: int = 4      # number of chunks returned per query

    # ------------------------------------------------------------------ #
    # Storage paths                                                        #
    # ------------------------------------------------------------------ #
    UPLOAD_DIR: str = "./uploaded_docs"

    # ------------------------------------------------------------------ #
    # Pydantic-settings: load .env automatically                          #
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,        # LLM_PROVIDER ≠ llm_provider
        extra="ignore",             # silently ignore unknown env vars
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Using @lru_cache means the .env file is parsed only once per process
    and the same object is reused everywhere — important for performance
    and for predictable behaviour during testing (override via
    dependency injection or monkeypatching).
    """
    return Settings()


# Convenience alias used throughout the codebase:
#   from app.config import settings
settings: Settings = get_settings()
