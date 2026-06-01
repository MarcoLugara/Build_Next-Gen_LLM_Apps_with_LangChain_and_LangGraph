# app/models.py

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""

    query: str = Field(...,
                 description="User's question or prompt", min_length=1)
    context: Optional[str] = Field(default="",
                 description="Optional conversation context or system instructions")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is your return policy for defective items?", "context": ""},
                {"query": "How do I reset my router?",
                 "context": "Previous conversation: customer has a router model XR-100"}
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for the /chat endpoint on success."""

    answer: str = Field(...,
                 description="LLM-generated or cached answer")
    cache_hit: bool = Field(...,
                 description="Whether response came from cache (exact or semantic)")
    cache_type: Optional[str] = Field(default=None,
                 description="Which cache provided the hit: 'exact', 'semantic', or None")
    tokens_sent: int = Field(..., ge=0,
                 description="Number of tokens in the prompt sent to LLM (0 if cache hit)")
    tokens_received: int = Field(..., ge=0,
                 description="Number of tokens in the answer (0 if cache hit)")
    tokens_saved: int = Field(..., ge=0,
                 description="Tokens saved by cache hit (0 if cache miss)")
    latency_ms: float = Field(..., ge=0,
                 description="Total request latency in milliseconds")
    truncated: bool = Field(default=False,
                 description="Whether prompt was truncated (always false in Alert&Reject strategy)")
    truncated_warning: Optional[str] = Field(default=None,
                 description="Warning message if truncation would be needed (reserved for future)")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(...,
                 description="Error code or type")
    message: str = Field(...,
                 description="Human-readable error description")
    details: Optional[dict] = Field(default=None, description="Additional error details")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"error": "TOKEN_LIMIT_EXCEEDED",
                 "message": "Prompt token count (8500) exceeds limit (7000). Please shorten your query.",
                 "details": {"token_count": 8500, "limit": 7000}}
            ]
        }
    }


class CacheStatsResponse(BaseModel):
    """Response model for /cache/stats endpoint (optional)."""

    exact_cache_size: int = Field(..., ge=0,
                description="Number of entries in Redis exact cache")
    semantic_cache_size: int = Field(..., ge=0,
                description="Number of entries in Chroma semantic cache")
    semantic_cache_max_entries: int = Field(..., ge=0,
                description="Maximum configured entries for semantic cache (LRU limit)")
    cache_hit_rate_total: float = Field(..., ge=0.0, le=1.0,
                description="Overall cache hit rate (exact + semantic) since startup")
    cache_hit_rate_exact: float = Field(..., ge=0.0, le=1.0,
                description="Exact cache hit rate")
    cache_hit_rate_semantic: float = Field(..., ge=0.0, le=1.0,
                description="Semantic cache hit rate")
    total_requests: int = Field(..., ge=0,
                description="Total requests processed")
    total_tokens_saved: int = Field(..., ge=0,
                description="Accumulated tokens saved by cache hits")