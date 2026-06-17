# tests/test_semantic.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.cache.semantic import SemanticCache    #the original class we are testing
from app.config import settings

@pytest.mark.asyncio
async def test_semantic_cache_get_hit(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_chroma_collection.query.return_value = {
        'ids': [['doc123']],
        'distances': [[0.05]],
        'metadatas': [[{
            'query': 'original query',
            'response': 'cached answer'
        }]]
    }
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                with patch.object(settings, 'similarity_threshold', 0.9):
                    response, cached_query, sim = await cache.get("test query", "")
                    assert response == "cached answer"
                    assert cached_query == "original query"
                    assert sim == 0.95
                    mock_redis_client.zadd.assert_called_once()

@pytest.mark.asyncio
async def test_semantic_cache_get_miss(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_chroma_collection.query.return_value = {
        'ids': [],
        'distances': [[]],
        'metadatas': [[]]
    }
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                response, cached_query, sim = await cache.get("test", "")
                assert response is None
                assert cached_query is None
                assert sim is None
                mock_redis_client.zadd.assert_not_called()

@pytest.mark.asyncio
async def test_semantic_cache_add(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [0.5, 0.6]
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                await cache.add("query", "context", "response text")
                mock_chroma_collection.add.assert_called_once()
                mock_redis_client.zadd.assert_called_once()
                add_args = mock_chroma_collection.add.call_args[1]
                assert 'metadatas' in add_args
                metadata = add_args['metadatas'][0]
                assert 'query' in metadata
                assert 'response' in metadata
                assert 'created_at' in metadata

@pytest.mark.asyncio
async def test_semantic_cache_eviction(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value.tolist.return_value = [1.0]
    mock_redis_client.zcard = AsyncMock(return_value=15000)
    mock_redis_client.zpopmin = AsyncMock(return_value=[('doc_to_evict', 123456789)])
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                with patch.object(settings, 'max_semantic_cache_entries', 10000):
                    with patch.object(settings, 'semantic_cache_lru_check_frequency', 1):
                        await cache.add("q", "c", "r")
                        mock_redis_client.zpopmin.assert_called_once()
                        mock_chroma_collection.delete.assert_called_once_with(ids=['doc_to_evict'])

@pytest.mark.asyncio
async def test_semantic_cache_stats(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_chroma_collection.count.return_value = 123
    mock_redis_client.zcard = AsyncMock(return_value=123)
    mock_embedder = MagicMock()
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                stats = await cache.stats()
                assert stats['semantic_cache_entries_chroma'] == 123
                assert stats['semantic_cache_entries_redis'] == 123
                assert stats['consistent'] is True

@pytest.mark.asyncio
async def test_semantic_cache_stats_inconsistency(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_chroma_collection.count.return_value = 100
    mock_redis_client.zcard = AsyncMock(return_value=120)
    mock_embedder = MagicMock()
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                stats = await cache.stats()
                assert stats['consistent'] is False

@pytest.mark.asyncio
async def test_semantic_cache_close(mock_redis_client):
    mock_chroma_collection = AsyncMock()
    mock_embedder = MagicMock()
    with patch('app.cache.semantic.chromadb.PersistentClient') as mock_chroma_client:
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_chroma_collection
        with patch('app.cache.semantic.SentenceTransformer', return_value=mock_embedder):
            with patch('app.cache.semantic.redis.from_url', return_value=mock_redis_client):
                cache = SemanticCache()
                mock_chroma_client.return_value.close = MagicMock()
                await cache.close()
                mock_redis_client.close.assert_called_once()
                mock_chroma_client.return_value.close.assert_called_once()