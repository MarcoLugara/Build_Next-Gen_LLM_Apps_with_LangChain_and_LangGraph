# app/main.py

import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.models import ChatRequest, ChatResponse, ErrorResponse, CacheStatsResponse
from app.cache.exact import ExactCache
from app.cache.semantic import SemanticCache
from app.cache.token_control import TokenValidator, TokenLimitExceededError
from app.llm_client import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Semantic Cache Proxy...")
    app.state.exact_cache = ExactCache()
    app.state.semantic_cache = SemanticCache()
    app.state.token_validator = TokenValidator()
    app.state.llm_client = LLMClient()
    logger.info("All components initialised. Ready to serve requests.")
    yield
    # Shutdown
    logger.info("Shutting down...")
    await app.state.exact_cache.close()
    await app.state.semantic_cache.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Semantic Cache Proxy",
    description="Production‑grade LLM cache with exact and semantic layers, token control, and LRU eviction.",
    version="1.0.0",
    lifespan=lifespan
)


def build_prompt(query: str, context: str) -> str:
    if context:
        return f"Context: {context}\n\nUser: {query}"
    else:
        return f"User: {query}"


@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def chat(request: ChatRequest):
    start_time = time.time()
    query = request.query
    context = request.context or ""

    # 1. Exact cache
    exact_response = await app.state.exact_cache.get(query, context)
    if exact_response:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Exact cache HIT for query: {query[:50]}...")
        return ChatResponse(
            answer=exact_response,
            cache_hit=True,
            cache_type="exact",
            tokens_sent=0,
            tokens_received=0,
            tokens_saved=0,
            latency_ms=round(latency_ms, 2),
            truncated=False,
            truncated_warning=None
        )

    # 2. Semantic cache
    semantic_response, cached_query, similarity = await app.state.semantic_cache.get(query, context)
    if semantic_response:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Semantic cache HIT for query: {query[:50]}... (similarity={similarity:.3f})")
        return ChatResponse(
            answer=semantic_response,
            cache_hit=True,
            cache_type="semantic",
            tokens_sent=0,
            tokens_received=0,
            tokens_saved=0,
            latency_ms=round(latency_ms, 2),
            truncated=False,
            truncated_warning=None
        )

    # 3. Cache miss – build prompt and apply token control
    prompt = build_prompt(query, context)
    try:
        processed_prompt, token_metadata = await app.state.token_validator.prepare_prompt(prompt)
    except TokenLimitExceededError as e:
        logger.warning(f"Token limit exceeded for prompt: {e}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="TOKEN_LIMIT_EXCEEDED",
                message=str(e),
                details={"token_count": e.token_count, "limit": e.max_tokens}
            ).model_dump()
        )

    # 4. Call LLM
    try:
        llm_response = await app.state.llm_client.generate(processed_prompt)
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="LLM_SERVICE_UNAVAILABLE",
                message="The LLM service is currently unavailable. Please try again later.",
                details={"exception": str(e)}
            ).model_dump()
        )

    # 5. Store in caches (fire-and-forget)
    asyncio.create_task(app.state.exact_cache.set(query, context, llm_response))
    asyncio.create_task(app.state.semantic_cache.add(query, context, llm_response))

    # 6. Prepare response
    latency_ms = (time.time() - start_time) * 1000
    prompt_tokens = token_metadata.get("original_token_count", 0)
    response_tokens = app.state.token_validator.count_tokens(llm_response)
    tokens_sent = prompt_tokens
    tokens_received = response_tokens
    tokens_saved = 0

    logger.info(f"Cache miss – LLM generated response in {latency_ms:.2f}ms (prompt_tokens={prompt_tokens}, response_tokens={response_tokens})")

    return ChatResponse(
        answer=llm_response,
        cache_hit=False,
        cache_type=None,
        tokens_sent=tokens_sent,
        tokens_received=tokens_received,
        tokens_saved=tokens_saved,
        latency_ms=round(latency_ms, 2),
        truncated=token_metadata.get("truncated", False),
        truncated_warning=token_metadata.get("warning")
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    exact_stats = await app.state.exact_cache.stats()
    semantic_stats = await app.state.semantic_cache.stats()
    return CacheStatsResponse(
        exact_cache_size=exact_stats.get("exact_cache_entries", 0),
        semantic_cache_size=semantic_stats.get("semantic_cache_entries_chroma", 0),
        semantic_cache_max_entries=semantic_stats.get("max_entries", settings.max_semantic_cache_entries),
        cache_hit_rate_total=0.0,
        cache_hit_rate_exact=0.0,
        cache_hit_rate_semantic=0.0,
        total_requests=0,
        total_tokens_saved=0
    )


@app.get("/health")
async def health():
    return {"status": "ok"}