# app/cache/exact.py

import hashlib
from typing import Optional
import redis.asyncio as redis
from loguru import logger
from app.config import settings


class ExactCache:
    """Redis-based exact cache for identical query+context pairs."""

    def __init__(self):
        self.redis = redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True
        )
        logger.info(f"ExactCache initialized with Redis URL: {settings.redis_url}")

    def _make_key(self, query: str, context: str) -> str:
        """Create a deterministic cache key from query and context."""
        content = f"{context}|{query}"
        hash_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"chat:{hash_hex}"

    async def get(self, query: str, context: str = "") -> Optional[str]:
        """Retrieve cached response for exact query+context."""
        key = self._make_key(query, context)
        response = await self.redis.get(key)
        if response is not None:
            logger.debug(f"Exact cache HIT for key {key[:20]}...")
        else:
            logger.debug(f"Exact cache MISS for key {key[:20]}...")
        return response

    async def set(self, query: str, context: str, response: str) -> None:
        """Store response in exact cache with TTL from config."""
        key = self._make_key(query, context)
        await self.redis.set(key, response, ex=settings.redis_ttl_seconds)
        logger.debug(f"Exact cache SET for key {key[:20]}... TTL={settings.redis_ttl_seconds}s")

    async def delete(self, query: str, context: str = "") -> bool:
        """Manually remove an entry from exact cache."""
        key = self._make_key(query, context)
        deleted_count = await self.redis.delete(key)
        if deleted_count:
            logger.info(f"Exact cache DELETED key {key[:20]}...")
        else:
            logger.debug(f"Exact cache DELETE attempted on non-existent key {key[:20]}...")
        return bool(deleted_count)

    async def stats(self) -> dict:
        """Return approximate number of exact cache entries."""
        count = 0
        async for key in self.redis.scan_iter(match="chat:*", count=1000):
            count += 1
        return {"exact_cache_entries": count}

    async def close(self):
        """Close Redis connection pool."""
        await self.redis.close()
        logger.info("ExactCache Redis connection closed")