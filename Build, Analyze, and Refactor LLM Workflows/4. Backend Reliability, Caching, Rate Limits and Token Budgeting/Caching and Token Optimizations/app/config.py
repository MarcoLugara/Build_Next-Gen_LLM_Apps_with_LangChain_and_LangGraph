# app/config.py

from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Groq API
    groq_api_key: str = Field(..., description="Groq API key (required)")
    groq_model: str = Field(default="mixtral-8x7b-32768",
                                   description="Groq model to use")
    groq_temperature: float = Field(default=0.0, ge=0.0, le=1.0,
                                   description="LLM temperature")
    groq_max_tokens: int = Field(default=1024, ge=1,
                                   description="Max output tokens")

    # Redis (exact cache + LRU coordination)
    redis_url: str = Field(default="redis://localhost:6379",
                                   description="Redis connection URL")
    redis_ttl_seconds: int = Field(default=3600, ge=1,
                                   description="TTL for exact cache entries (seconds)")
    redis_max_connections: int = Field(default=10, ge=1,
                                   description="Redis connection pool size")

    # Semantic cache (Chroma)
    chroma_persist_directory: str = Field(default="./chroma_cache",
                                   description="Directory for Chroma persistent storage")
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0,
                                   description="Cosine similarity threshold for cache hit")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2",
                                   description="Sentence transformer for embeddings")

    #LRU eviction for semantic cache (Chroma)
    max_semantic_cache_entries: int = Field(default=10000, ge=1,
                                   description="Maximum number of semantic cache entries (LRU eviction)")
    semantic_cache_lru_check_frequency: int = Field(default=100, ge=1,
                                   description="Check LRU size every N insertions")

    # Token control
    max_prompt_tokens: int = Field(default=7000, ge=100,
                                   description="Maximum tokens allowed in prompt (input + context)")
    token_encoder_name: str = Field(default="cl100k_base",
                                   description="tiktoken encoder name")
    reject_on_overflow: bool = Field(default=True,
                                   description="If True, reject request when prompt exceeds max_prompt_tokens; if False, future extensions")

    # Performance
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, description="HTTP timeout for Groq API")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retries on transient LLM failures")
    retry_backoff_factor: float = Field(default=1.0, ge=0.1, description="Exponential backoff factor")

    # Logging
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()