# tests/test_llm_client.py
import pytest
from unittest.mock import AsyncMock, patch
from httpx import NetworkError, TimeoutException
from app.llm_client import LLMClient        #the original class we are testing

@pytest.mark.asyncio
async def test_llm_client_generate_success():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AsyncMock(content="Mocked response")
    with patch('app.llm_client.ChatGroq', return_value=mock_llm):
        client = LLMClient()
        result = await client.generate("Hello")
        assert result == "Mocked response"
        mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_llm_client_retry_on_network_error():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = [NetworkError("Fake network error"), AsyncMock(content="Success after retry")]
    with patch('app.llm_client.ChatGroq', return_value=mock_llm):
        client = LLMClient()
        prompt = "Retry test"
        result = await client.generate(prompt)
        assert result == "Success after retry"
        assert mock_llm.ainvoke.call_count == 2

@pytest.mark.asyncio
async def test_llm_client_retry_on_timeout():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = [TimeoutException("Timeout"), AsyncMock(content="Success")]
    with patch('app.llm_client.ChatGroq', return_value=mock_llm):
        client = LLMClient()
        prompt = "Timeout test"
        result = await client.generate(prompt)
        assert result == "Success"
        assert mock_llm.ainvoke.call_count == 2

@pytest.mark.asyncio
async def test_llm_client_no_retry_on_other_exceptions():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = ValueError("Bad request")
    with patch('app.llm_client.ChatGroq', return_value=mock_llm):
        client = LLMClient()
        prompt = "Unknown error test"
        with pytest.raises(ValueError):
            await client.generate(prompt)
        assert mock_llm.ainvoke.call_count == 1

@pytest.mark.asyncio
async def test_llm_client_max_retries_exhausted():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = NetworkError("Persistent failure")
    with patch('app.llm_client.ChatGroq', return_value=mock_llm):
        client = LLMClient()
        prompt = "It will retry 3 times then fail"
        with pytest.raises(NetworkError):
            await client.generate(prompt)
        assert mock_llm.ainvoke.call_count == 3