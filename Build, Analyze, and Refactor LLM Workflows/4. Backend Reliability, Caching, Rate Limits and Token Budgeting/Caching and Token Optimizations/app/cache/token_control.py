# app/cache/token_control.py

import tiktoken
from loguru import logger
from app.config import settings


class TokenLimitExceededError(Exception):
    def __init__(self, token_count: int, max_tokens: int):
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(f"Prompt token count ({token_count}) exceeds limit ({max_tokens})")


class TokenValidator:
    def __init__(self):
        self.encoder = tiktoken.get_encoding(settings.token_encoder_name)
        logger.info(f"TokenValidator initialised with encoder: {settings.token_encoder_name}")

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _truncate_sliding_window(self, prompt: str) -> tuple[str, dict]:
        original_tokens = self.count_tokens(prompt)
        if original_tokens <= settings.max_prompt_tokens:
            return prompt, {"truncated": False, "original_token_count": original_tokens}

        keep_start = int(settings.max_prompt_tokens * settings.truncation_keep_start_ratio)
        keep_end = settings.max_prompt_tokens - keep_start
        tokens = self.encoder.encode(prompt)
        truncated_tokens = tokens[:keep_start] + (tokens[-keep_end:] if keep_end > 0 else [])
        truncated_prompt = self.encoder.decode(truncated_tokens)
        truncated_count = len(truncated_tokens)

        logger.info(f"Truncated prompt from {original_tokens} to {truncated_count} tokens (sliding_window)")
        return truncated_prompt, {
            "truncated": True,
            "original_token_count": original_tokens,
            "truncated_token_count": truncated_count,
            "strategy": "truncate_with_warning",
            "warning": "Prompt was automatically truncated: the middle part was removed to fit token limit."
        }

    def prepare_prompt(self, prompt: str) -> tuple[str, dict]:
        token_count = self.count_tokens(prompt)
        if token_count <= settings.max_prompt_tokens:
            return prompt, {"truncated": False, "original_token_count": token_count}

        strategy = settings.overflow_strategy
        if strategy == "reject":
            logger.warning(f"Token limit exceeded: {token_count} > {settings.max_prompt_tokens} – rejecting")
            raise TokenLimitExceededError(token_count, settings.max_prompt_tokens)
        elif strategy == "truncate_with_warning":
            return self._truncate_sliding_window(prompt)
        else:
            logger.warning(f"Unknown overflow strategy '{strategy}', falling back to 'reject'")
            raise TokenLimitExceededError(token_count, settings.max_prompt_tokens)