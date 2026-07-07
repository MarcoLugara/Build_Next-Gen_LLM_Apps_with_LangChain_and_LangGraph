from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ChatRequest(BaseModel):
    """Incoming chat request from a client."""

    query: str = Field(..., description="The user's input/question", min_length=1)
    context: Optional[str] = Field(default=None, description="Optional prior conversation history or system context")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier for multi-tenant isolation and budgeting")
    provider_key: Optional[str] = Field(default=None, description="Force a specific provider (e.g., 'primary', 'fallback')")
    model_name: Optional[str] = Field(default=None, description="Force a specific model (e.g., 'gpt-4-turbo')")
    max_tokens_override: Optional[int] = Field(default=None, ge=1, description="Override per-request max_tokens")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata for tracing/debugging")

    @field_validator("provider_key")
    @classmethod
    def validate_provider_key(cls, v):
        if v is not None and not v.strip():
            raise ValueError("provider_key must be a non-empty string if provided")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("model_name must be a non-empty string if provided")
        return v


class BudgetStatus(BaseModel):
    """FinOps budget status for a tenant."""

    tenant_id: str = Field(..., description="Tenant identifier")
    budget_usd: float = Field(..., ge=0.0, description="Total budget allocated for this tenant")
    spent_usd: float = Field(..., ge=0.0, description="Total spent in the current budget window")
    remaining_usd: float = Field(..., ge=0.0, description="Remaining budget")
    spent_ratio: float = Field(..., ge=0.0, le=1.0, description="Fraction of budget spent (0.0 to 1.0)")
    is_warning: bool = Field(default=False, description="True if spent_ratio >= warning_threshold_ratio")
    is_hard_limit_reached: bool = Field(default=False, description="True if spent_ratio >= hard_limit_ratio")
    downgrade_applied: bool = Field(default=False, description="True if auto-downgrade is active for this tenant")
    current_model: Optional[str] = Field(default=None, description="The model currently serving this tenant (may differ from requested)")


class EvalMetadata(BaseModel):
    """Evaluation metrics attached to a response (from continuous evaluation engine)."""

    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Overall quality score (0-1)")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Individual metrics (e.g., {'answer_relevancy': 0.92, 'faithfulness': 0.88})")
    hallucination_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Hallucination probability (0=no hallucination, 1=highly hallucinated)")
    evaluated_at: Optional[datetime] = Field(default=None, description="Timestamp when evaluation was performed")


class ChatResponse(BaseModel):
    """Response to a chat request, including answer and comprehensive metadata."""

    query: str = Field(..., description="Original query (may be truncated/summarized if token control applied)")
    response: str = Field(..., description="The generated answer from the LLM")
    provider_key: str = Field(..., description="Which provider was used (e.g., 'primary')")
    model_name: str = Field(..., description="Which model was used (e.g., 'gpt-4-turbo')")
    cache_hit: bool = Field(default=False, description="True if response came from cache (exact or semantic)")
    cache_type: Optional[Literal["exact", "semantic"]] = Field(default=None, description="Which cache layer provided the hit")
    cached_query: Optional[str] = Field(default=None, description="The original query that was cached (if semantic hit)")
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Cosine similarity score (if semantic cache hit)")
    token_control_applied: bool = Field(default=False, description="True if token control modified the prompt (truncate/summarize)")
    token_control_strategy: Optional[Literal["reject", "truncate_with_warning", "summarize_overflow"]] = Field(
        default=None, description="Which token control strategy was used"
    )
    token_control_warning: Optional[str] = Field(default=None, description="Warning message from token control (e.g., 'Prompt truncated')")
    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the actual prompt sent to the LLM")
    response_tokens: int = Field(..., ge=0, description="Number of tokens in the LLM response")
    total_tokens: int = Field(..., ge=0, description="Total tokens used (prompt + response)")
    tokens_saved: int = Field(default=0, ge=0, description="Tokens saved due to cache hit (0 if miss)")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD for this request")
    budget_status: Optional[BudgetStatus] = Field(default=None, description="Budget status for the tenant (if multi-tenant)")
    eval_metadata: Optional[EvalMetadata] = Field(default=None, description="Quality evaluation scores (if available)")
    latency_ms: float = Field(..., ge=0.0, description="End-to-end latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp (UTC)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata from the request (propagated)")


class ErrorResponse(BaseModel):
    """Structured error response."""

    error_type: str = Field(..., description="Machine-readable error type (e.g., 'TOKEN_LIMIT_EXCEEDED', 'LLM_SERVICE_UNAVAILABLE')")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional structured details (e.g., token counts, budget info)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp (UTC)")


class CacheStatsResponse(BaseModel):
    """Cache statistics for monitoring."""

    exact_cache_size: int = Field(..., ge=0, description="Number of keys in Redis exact cache")
    semantic_cache_size: int = Field(..., ge=0, description="Number of entries in Chroma semantic cache")
    semantic_cache_max_entries: int = Field(..., ge=1, description="Max entries before LRU eviction (from config)")
    total_requests: int = Field(..., ge=0, description="Total number of requests processed (since startup or reset)")
    cache_hit_rate_total: float = Field(..., ge=0.0, le=1.0, description="Overall cache hit rate (exact + semantic)")
    cache_hit_rate_exact: float = Field(..., ge=0.0, le=1.0, description="Exact cache hit rate")
    cache_hit_rate_semantic: float = Field(..., ge=0.0, le=1.0, description="Semantic cache hit rate")
    total_tokens_saved: int = Field(..., ge=0, description="Total tokens saved by cache hits")


class ServiceStatus(BaseModel):
    """Health check response for the gateway service."""

    service_name: str = Field(
        ...,
        description="Service name for telemetry and health checks (from settings.observability.service_name)"
    )
    status: Literal["ok", "degraded", "unavailable"] = Field(..., description="Overall health status")
    startup_timestamp: datetime = Field(..., description="UTC timestamp when the service started")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds (calculated from startup_timestamp)")
    dependencies: Dict[str, Literal["healthy", "unhealthy", "unknown"]] = Field(
        default_factory=dict,
        description="Health status of each dependency (e.g., redis, chroma, groq, openai)"
    )