# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app     #the original class we are testing
from app.cache.token_control import TokenLimitExceededError      #for the rejection in case of TokenLimitExceededError

@pytest.fixture
def client():
    with patch('app.main.ExactCache') as mock_exact_cache_cls:
        with patch('app.main.SemanticCache') as mock_semantic_cache_cls:
            with patch('app.main.TokenValidator') as mock_token_validator_cls:
                with patch('app.main.LLMClient') as mock_llm_client_cls:
                    mock_exact_cache = AsyncMock()
                    mock_semantic_cache = AsyncMock()
                    mock_token_validator = AsyncMock()
                    mock_llm_client = AsyncMock()
                    mock_exact_cache_cls.return_value = mock_exact_cache
                    mock_semantic_cache_cls.return_value = mock_semantic_cache
                    mock_token_validator_cls.return_value = mock_token_validator
                    mock_llm_client_cls.return_value = mock_llm_client
                    app.state.exact_cache = mock_exact_cache
                    app.state.semantic_cache = mock_semantic_cache
                    app.state.token_validator = mock_token_validator
                    app.state.llm_client = mock_llm_client
                    yield TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_cache_stats_endpoint(client):
    client.app.state.exact_cache.stats = AsyncMock(return_value={"exact_cache_entries": 5})
    client.app.state.semantic_cache.stats = AsyncMock(return_value={
        "semantic_cache_entries_chroma": 10,
        "max_entries": 10000
    })
    response = client.get("/cache/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["exact_cache_size"] == 5
    assert data["semantic_cache_size"] == 10
    assert data["semantic_cache_max_entries"] == 10000

def test_chat_endpoint_exact_cache_hit(client):
    client.app.state.exact_cache.get = AsyncMock(return_value="exact answer")
    client.app.state.semantic_cache.get = AsyncMock(return_value=(None, None, None))
    response = client.post("/chat", json={"query": "test", "context": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["cache_hit"] is True
    assert data["cache_type"] == "exact"
    assert data["answer"] == "exact answer"

def test_chat_endpoint_semantic_cache_hit(client):
    client.app.state.exact_cache.get = AsyncMock(return_value=None)
    client.app.state.semantic_cache.get = AsyncMock(return_value=("semantic answer", "cached query", 0.95))
    response = client.post("/chat", json={"query": "similar question", "context": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["cache_hit"] is True
    assert data["cache_type"] == "semantic"
    assert data["answer"] == "semantic answer"

def test_chat_endpoint_cache_miss(client):
    client.app.state.exact_cache.get = AsyncMock(return_value=None)
    client.app.state.semantic_cache.get = AsyncMock(return_value=(None, None, None))
    client.app.state.token_validator.prepare_prompt = AsyncMock(return_value=("processed prompt", {"truncated": False, "original_token_count": 50}))
    client.app.state.llm_client.generate = AsyncMock(return_value="LLM generated answer")
    client.app.state.token_validator.count_tokens = AsyncMock(return_value=100)
    response = client.post("/chat", json={"query": "new question", "context": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["cache_hit"] is False
    assert data["cache_type"] is None
    assert data["answer"] == "LLM generated answer"
    assert data["tokens_sent"] == 50
    assert data["tokens_received"] == 100
    client.app.state.exact_cache.set.assert_called_once()
    client.app.state.semantic_cache.add.assert_called_once()

def test_chat_endpoint_token_rejection(client):
    client.app.state.token_validator.prepare_prompt = AsyncMock(side_effect=TokenLimitExceededError(8500, 7000))
    response = client.post("/chat", json={"query": "old_man_talking" * 1000, "context": ""})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "TOKEN_LIMIT_EXCEEDED"
    assert data["details"]["token_count"] == 8500
    assert data["details"]["limit"] == 7000

def test_chat_endpoint_llm_service_unavailable(client):
    client.app.state.exact_cache.get = AsyncMock(return_value=None)
    client.app.state.semantic_cache.get = AsyncMock(return_value=(None, None, None))
    client.app.state.token_validator.prepare_prompt = AsyncMock(return_value=("prompt", {}))
    client.app.state.llm_client.generate = AsyncMock(side_effect=Exception("LLM connection error"))
    response = client.post("/chat", json={"query": "test", "context": ""})
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "LLM_SERVICE_UNAVAILABLE"
    assert "LLM connection error" in data["message"]