import os
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator   #data models, metadata and constraints, validation logics
from pydantic_settings import BaseSettings    #reading .env files


class ModelConfig(BaseModel):
    """Custom model settings for a specific LLM."""

    model_name: str = Field(...,
        description="Model identifier (e.g., 'gpt-4-turbo', 'llama3-70b-8192')"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Sampling temperature for generation"
    )
    max_tokens: int = Field(
        default=4096, ge=1,
        description="Maximum tokens to generate per response"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Nucleus sampling probability"
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Penalize repeated tokens"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0,
        description="Penalize new topic tokens"
    )

class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider to switch between them."""

    provider_type: Literal["groq", "openai", "anthropic", "local"] = Field(...,
        description="Provider identifier used for routing"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for this provider (can be None for local)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL (e.g., for local proxies)"
    )
    models: Dict[str, ModelConfig] = Field(
        default_factory=dict,  #if default, maps to empty dictionary
        description="Linking of model_name with the ModelConfig for available models under this provider"
    )
    default_model: Optional[str] = Field(
        default=None,
        description="Default model to use if none specified"
    )

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, v, info):
        if v is not None and v not in info.data.get("models", {}):
            raise ValueError(f"default_model '{v}' must be one of the keys in models")
        return v

class CacheConfig(BaseModel):
    """Configuration for the exact (Redis) and semantic (Chroma) caches."""

    # Redis (exact cache)
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    redis_ttl_seconds: int = Field(
        default=3600, ge=1,
        description="TTL for exact cache entries"
    )
    redis_max_connections: int = Field(
        default=10, ge=1,
        description="Max Redis connection pool size"
    )
    redis_key_prefix: str = Field(
        default="fabric:chat:",
        description="Prefix for all Redis keys"
    )

    # Chroma (semantic cache)
    chroma_persist_directory: str = Field(
        default="./chroma_cache",
        description="Directory for Chroma persistent storage"
    )
    similarity_threshold: float = Field(
        default=0.92, ge=0.0, le=1.0,
        description="Min cosine similarity for cache hit"
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model for embeddings"
    )
    max_semantic_cache_entries: int = Field(
        default=10000, ge=1,
        description="Max entries before LRU eviction"
    )
    semantic_cache_lru_check_frequency: int = Field(
        default=10, ge=1,
        description="Check LRU every N inserts"
    )

    # Optional for scalability: multi-tenant isolation
    tenant_key_prefix: Optional[str] = Field(
        default=None,
        description="If set, keys will be namespaced per tenant: {tenant_key_prefix}:{tenant_id}:..."
    )

class TokenConfig(BaseModel):
    """Configuration for token counting and overflow strategies."""

    tokenizer_model_name: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace tokenizer name (must match the main LLM's tokenizer)"
    )
    max_prompt_tokens: int = Field(
        default=7000, ge=1,
        description="Maximum allowed tokens in the prompt"
    )
    overflow_strategy: Literal["reject", "truncate_with_warning", "summarize_overflow"] = Field(
        default="reject",
        description="What to do when max_prompt_tokens is exceeded"
    )
    truncation_keep_start_ratio: float = Field(
        default=0.6, ge=0.1, le=0.9,
        description="Fraction of tokens to keep from the start (truncate strategy)"
    )
    summarization_keep_ratio: float = Field(
        default=0.6, ge=0.1, le=0.9,
        description="Fraction of prompt to keep before summarizing"
    )
    summarization_max_tokens: int = Field(
        default=512, ge=10,
        description="Max tokens for the summary output"
    )

class BudgetConfig(BaseModel):
    """FinOps configuration: budgets per tenant, alert thresholds, and auto-downgrade rules."""

    budget_window_days: int = Field(
        default=30, ge=1,
        description="Budget window in days (e.g., monthly = 30)"
    )
    default_tenant_budget_usd: float = Field(
        default=100.0, ge=0.0,
        description="Default budget per tenant"
    )
    warning_threshold_ratio: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Warn at X% of budget spent"
    )
    hard_limit_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Reject at X% of budget (default 100%)"
    )
    auto_downgrade_enabled: bool = Field(
        default=False,
        description="If True, downgrade to cheaper model when budget exceeds warning threshold"
    )
    downgrade_model_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping: original_model -> fallback_model (e.g., 'gpt-4-turbo' -> 'gpt-3.5-turbo')"
    )
    cost_per_1k_tokens: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost per 1000 tokens for each (provider,model) pair. Key format: 'provider:model'"
    )

class EvalConfig(BaseModel):
    """Configuration for the continuous evaluation engine (RAGAS, hallucination, drift)."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable background evaluation"
    )
    golden_dataset_path: Optional[str] = Field(
        default=None,
        description="Path to a JSON/CSV file with golden Q&A pairs for accuracy evaluation"
    )
    eval_frequency_seconds: int = Field(
        default=3600, ge=10,
        description="Run evaluation pipeline every N seconds"
    )
    metrics: List[str] = Field(
        default=["answer_relevancy", "faithfulness", "context_precision"],
        description="List of RAGAS/DeepEval metrics to compute"
    )
    hallucination_model_name: str = Field(
        default="vectara/hallucination_evaluation_model",
        description="Model used for hallucination detection (usually a small classifier)"
    )
    drift_detection_window_minutes: int = Field(
        default=60, ge=5,
        description="Window (in minutes) to compute query embedding drift"
    )

class ObservabilityConfig(BaseModel):
    """OpenTelemetry, Prometheus, and logging configuration."""

    service_name: str = Field(
        default="ai-fabric-gateway",
        description="Service name for telemetry"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint"
    )
    metrics_port: int = Field(
        default=8001, ge=1024, le=65535,
        description="Port for metrics server"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application log level"
    )
    log_json: bool = Field(
        default=False,
        description="Output logs in JSON format"
    )

class ResilienceConfig(BaseModel):
    """Timeout, retry, and circuit breaker settings."""

    request_timeout_seconds: float = Field(
        default=30.0, ge=1.0,
        description="HTTP request timeout for LLM calls"
    )
    retry_attempts: int = Field(
        default=3, ge=0,
        description="Number of retry attempts for LLM calls"
    )
    retry_backoff_factor: float = Field(
        default=1.0, ge=0.1,
        description="Exponential backoff multiplier"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1,
        description="Failures before circuit opens"
    )
    circuit_breaker_recovery_timeout_seconds: int = Field(
        default=60, ge=1,
        description="Time before attempting reset"
    )
    circuit_breaker_success_threshold: int = Field(
        default=2, ge=1,
        description="Successful calls to close circuit again"
    )

class Settings(BaseSettings):
    """Main application settings. Loads from environment variables and .env files."""

    providers: Dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Dictionary of provider_key -> ProviderConfig. E.g., {'groq': ProviderConfig(...)}"
    )
    active_providers: List[str] = Field(
        default_factory=list,
        description="List of provider keys to use (order indicates routing priority)"
    )

    # Nested config classes
    cache: CacheConfig = Field(
        default_factory=CacheConfig
    )
    token: TokenConfig = Field(
        default_factory=TokenConfig
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig
    )
    eval: EvalConfig = Field(
        default_factory=EvalConfig
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    resilience: ResilienceConfig = Field(
        default_factory=ResilienceConfig
    )

    # Project metadata
    project_name: str = Field(
        default="AI Fabric Gateway",
        description="Name of the project"
    )
    environment: Literal["dev", "staging", "production"] = Field(
        default="dev", description="Deployment environment"
    )

    class Config:
        """Pydantic BaseSettings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    """Sanity check that the provider used exists"""
    @field_validator("active_providers")
    @classmethod
    def validate_active_providers(cls, v, info):
        if v:
            available = info.data.get("providers", {})
            missing = [p for p in v if p not in available]      #Create a new "missing" list. Take every provider p inside the active_providers list (v). If that provider p is NOT part of the keys of the available dictionary, add it to the list missing."
            if missing:
                raise ValueError(f"Active providers not found in 'providers' config: {missing}")
        return v

    def get_model_config(self, provider_key: str, model_name: Optional[str] = None) -> ModelConfig:
        """
        Sanity check and retriever for ModelConfig given provider and model.
        Returns clear errors in case something is missing.
        """
        provider = self.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Provider '{provider_key}' not found")
        model_name = model_name or provider.default_model
        if not model_name:
            raise ValueError(f"No model specified and no default_model set for provider '{provider_key}'")
        model = provider.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found for provider '{provider_key}'")
        return model

    def get_cost_per_1k_tokens(self, provider_key: str, model_name: str) -> float:
        """Retrieve cost per 1000 tokens for a specific (provider, model) pair."""
        key = f"{provider_key}:{model_name}"
        return self.budget.cost_per_1k_tokens.get(key, 0.0)


# For easy import
settings = Settings()