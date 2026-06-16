# tests/test_exact.py
import pytest
from unittest.mock import patch   #patch replaces redis.from_url with our mock
from app.cache.exact import ExactCache
from app.config import settings  #the real settings object

@pytest.mark.asyncio
async def test_exact_cache_make_key():
    with patch('app.cache.exact.redis') as mock_redis:
        cache = ExactCache()
        key = cache._make_key("test query", "context")
        assert key.startswith("chat:")
        assert len(key) == 64 + 5  # "chat:" + 64 hex chars

@pytest.mark.asyncio
async def test_exact_cache_get_hit(mock_redis_client):
    mock_redis_client.get.return_value = "cached response"
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        response = await cache.get("test", "")
        assert response == "cached response"
        mock_redis_client.get.assert_called_once()

@pytest.mark.asyncio
async def test_exact_cache_get_miss(mock_redis_client):
    mock_redis_client.get.return_value = None
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        response = await cache.get("test", "")
        assert response is None

@pytest.mark.asyncio
async def test_exact_cache_set(mock_redis_client):
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        await cache.set("query", "context", "response")
        mock_redis_client.set.assert_called_once()
        args, kwargs = mock_redis_client.set.call_args
        assert kwargs["ex"] == settings.redis_ttl_seconds

@pytest.mark.asyncio
async def test_exact_cache_delete(mock_redis_client):
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        result = await cache.delete("query", "context")
        assert result is True
        mock_redis_client.delete.assert_called_once()

@pytest.mark.asyncio
async def test_exact_cache_delete_not_found(mock_redis_client):
    mock_redis_client.delete.return_value = 0
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        result = await cache.delete("query", "context")
        assert result is False

@pytest.mark.asyncio
async def test_exact_cache_stats(mock_redis_client):
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        stats = await cache.stats()
        assert "exact_cache_entries" in stats
        assert stats["exact_cache_entries"] == 1

@pytest.mark.asyncio
async def test_exact_cache_close(mock_redis_client):
    with patch('app.cache.exact.redis.from_url', return_value=mock_redis_client):
        cache = ExactCache()
        await cache.close()
        mock_redis_client.close.assert_called_once()