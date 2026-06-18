# tests/test_token_control.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.cache.token_control import TokenValidator, TokenLimitExceededError
from app.config import settings

@pytest.fixture
def mock_tokenizer():
    mock = MagicMock()
    mock.encode.side_effect = lambda text: list(range(len(text)))
    mock.decode.side_effect = lambda tokens: ''.join(chr(ord('a')+t%26) for t in tokens)
    return mock

@pytest.mark.asyncio
async def test_prepare_prompt_under_limit(mock_tokenizer):
    with patch('app.cache.token_control.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        validator = TokenValidator()
        with patch.object(settings, 'max_prompt_tokens', 100):
            prompt = "short"
            processed, meta = await validator.prepare_prompt(prompt)
            assert processed == prompt
            assert meta['truncated'] is False
            assert meta['original_token_count'] == 5

@pytest.mark.asyncio
async def test_prepare_prompt_reject_strategy(mock_tokenizer):
    with patch('app.cache.token_control.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        validator = TokenValidator()
        with patch.object(settings, 'max_prompt_tokens', 10):
            with patch.object(settings, 'overflow_strategy', 'reject'):
                prompt = "this prompt is longer than ten tokens"
                with pytest.raises(TokenLimitExceededError) as exc:
                    await validator.prepare_prompt(prompt)
                assert exc.value.token_count > 10
                assert exc.value.max_tokens == 10

@pytest.mark.asyncio
async def test_prepare_prompt_truncate_with_warning(mock_tokenizer):
    with patch('app.cache.token_control.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        validator = TokenValidator()
        with patch.object(settings, 'max_prompt_tokens', 10):
            with patch.object(settings, 'overflow_strategy', 'truncate_with_warning'):
                with patch.object(settings, 'truncation_keep_start_ratio', 0.6):
                    prompt = "this is a longer prompt that needs truncation"
                    processed, meta = await validator.prepare_prompt(prompt)
                    assert meta['truncated'] is True
                    assert meta['strategy'] == 'truncate_with_warning'
                    assert 'warning' in meta
                    assert 'original_token_count' in meta
                    assert 'truncated_token_count' in meta
                    token_len = len(processed)
                    assert token_len <= 10

@pytest.mark.asyncio
async def test_prepare_prompt_summarize_overflow(mock_tokenizer):
    mock_summarizer = AsyncMock()
    mock_summarizer.ainvoke.return_value = MagicMock(content="Fake summary")
    with patch('app.cache.token_control.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        with patch('app.cache.token_control.ChatGroq', return_value=mock_summarizer):
            validator = TokenValidator()
            with patch.object(settings, 'max_prompt_tokens', 10):
                with patch.object(settings, 'overflow_strategy', 'summarize_overflow'):
                    with patch.object(settings, 'summarization_keep_ratio', 0.7):
                        prompt = "This is a very long prompt that will be summarised after keeping the first 70% of allowed tokens"
                        processed, meta = await validator.prepare_prompt(prompt)
                        assert meta['truncated'] is True
                        assert meta['strategy'] == 'summarize_overflow'
                        assert 'Fake summary' in processed
                        assert 'warning' in meta

@pytest.mark.asyncio
async def test_prepare_prompt_unknown_strategy_fallback(mock_tokenizer):
    with patch('app.cache.token_control.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        validator = TokenValidator()
        with patch.object(settings, 'max_prompt_tokens', 10):
            with patch.object(settings, 'overflow_strategy', 'unknown_strategy'):
                with pytest.raises(TokenLimitExceededError):
                    await validator.prepare_prompt("this prompt is long")