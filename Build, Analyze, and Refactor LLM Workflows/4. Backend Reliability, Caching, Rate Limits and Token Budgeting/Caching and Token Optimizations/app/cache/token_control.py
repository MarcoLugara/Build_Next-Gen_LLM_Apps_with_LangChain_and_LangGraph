# app/cache/token_control.py

import tiktoken
from loguru import logger
from app.config import settings


class TokenLimitExceededError(Exception):
    """Raised when prompt exceeds max_prompt_tokens and reject_on_overflow=True."""

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

    def validate(self, prompt: str) -> None:
        token_count = self.count_tokens(prompt)
        if token_count > settings.max_prompt_tokens:
            if settings.reject_on_overflow:
                logger.warning(f"Token limit exceeded: {token_count} > {settings.max_prompt_tokens}")
                raise TokenLimitExceededError(token_count, settings.max_prompt_tokens)
            else:
                logger.debug(f"Token limit exceeded but reject_on_overflow=False – will truncate")

    def truncate(self, prompt: str) -> tuple[str, dict]:
        original_tokens = self.count_tokens(prompt)
        if original_tokens <= settings.max_prompt_tokens:
            return prompt, {"truncated": False, "original_token_count": original_tokens}

        if settings.truncation_strategy != "sliding_window":
            logger.warning(
                f"Unknown truncation strategy '{settings.truncation_strategy}', falling back to sliding_window")

        keep_start = int(settings.max_prompt_tokens * settings.truncation_keep_start_ratio)
        keep_end = settings.max_prompt_tokens - keep_start

        tokens = self.encoder.encode(prompt)
        truncated_tokens = tokens[:keep_start] + (tokens[-keep_end:] if keep_end > 0 else [])
        truncated_prompt = self.encoder.decode(truncated_tokens)

        truncated_count = len(truncated_tokens)
        logger.info(f"Truncated prompt from {original_tokens} to {truncated_count} tokens (strategy=sliding_window)")

        return truncated_prompt, {
            "truncated": True,
            "original_token_count": original_tokens,
            "truncated_token_count": truncated_count,
            "strategy": "sliding_window"
        }

    def prepare_prompt(self, prompt: str) -> tuple[str, dict]:
        if settings.reject_on_overflow:
            self.validate(prompt)
            return prompt, {"truncated": False, "original_token_count": self.count_tokens(prompt)}
        else:
            return self.truncate(prompt)