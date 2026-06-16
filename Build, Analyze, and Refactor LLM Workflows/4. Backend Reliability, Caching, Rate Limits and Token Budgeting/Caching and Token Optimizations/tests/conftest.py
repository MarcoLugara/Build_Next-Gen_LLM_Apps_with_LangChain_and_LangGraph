# tests/conftest.py
import pytest       #testing framework
from unittest.mock import AsyncMock, MagicMock
from app.config import Settings

@pytest.fixture
def mock_settings():
    """Mock the global settings object to avoid reading .env during tests."""
    settings = MagicMock(spec=Settings)
    settings.redis_url = "redis://localhost:6379"
    settings.redis_ttl_seconds = 3600
    settings.redis_max_connections = 10
    settings.chroma_persist_directory = "./test_chroma"
    settings.similarity_threshold = 0.92
    settings.embedding_model_name = "all-MiniLM-L6-v2"
    settings.max_semantic_cache_entries = 10000
    settings.semantic_cache_lru_check_frequency = 100
    settings.max_prompt_tokens = 7000
    settings.tokenizer_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    settings.overflow_strategy = "reject"
    settings.truncation_keep_start_ratio = 0.6
    settings.summarization_model = "llama3-8b-8192"
    settings.summarization_max_tokens = 512
    settings.summarization_keep_ratio = 0.7
    settings.request_timeout_seconds = 30.0
    settings.retry_attempts = 3
    settings.retry_backoff_factor = 1.0
    settings.log_level = "DEBUG"
    return settings

@pytest.fixture
def mock_redis_client():
    """Return an AsyncMock for redis.asyncio client."""
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set.return_value = None
    mock.delete.return_value = 1
    async def scan_iter(*args, **kwargs):
        yield "chat:abc123"
    mock.scan_iter = scan_iter
    return mock

@pytest.fixture
def mock_chroma_collection():
    """Return a mock for Chroma collection."""
    mock = AsyncMock()
    mock.query.return_value = {
        'ids': [],
        'distances': [[]],
        'metadatas': [[]]
    }
    mock.add.return_value = None
    mock.count.return_value = 0
    mock.delete.return_value = None
    return mock