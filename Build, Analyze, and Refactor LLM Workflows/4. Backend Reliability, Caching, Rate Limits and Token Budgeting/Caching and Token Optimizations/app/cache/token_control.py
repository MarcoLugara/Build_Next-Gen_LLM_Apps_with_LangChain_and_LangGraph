# app/cache/token_control.py

import tiktoken
from langchain_groq import ChatGroq   #Used for the LLM summarization (cheap model)
from langchain_core.messages import HumanMessage  #Standard LangChain message format, allows customization
                                        #lanchain_core handles (backbone) message interactions between human and LLM
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

        self.summarizer_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.summarization_model,
            temperature=0.0,
            max_tokens=settings.summarization_max_tokens,
            timeout=settings.request_timeout_seconds,
        )
        logger.info(f"Summarisation LLM initialised with model: {settings.summarization_model}")

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

    async def _summarize_overflow(self, prompt: str) -> tuple[str, dict]:
        original_tokens = self.count_tokens(prompt)
        if original_tokens <= settings.max_prompt_tokens:
            return prompt, {"truncated": False, "original_token_count": original_tokens}

        keep_tokens = int(settings.max_prompt_tokens * 0.7)
        tokens = self.encoder.encode(prompt)
        kept_tokens = tokens[:keep_tokens]
        overflow_tokens = tokens[keep_tokens:]

        overflow_text = self.encoder.decode(overflow_tokens)
        summarise_prompt = f"Summarise the following text concisely, preserving key facts and information. Keep the summary under {settings.summarization_max_tokens} tokens.\n\nText:\n{overflow_text}"
        messages = [HumanMessage(content=summarise_prompt)]
        response = await self.summarizer_llm.ainvoke(messages)
        summary = response.content.strip()

        kept_part = self.encoder.decode(kept_tokens)
        processed_prompt = f"{kept_part}\n\n[Summary of omitted content]: {summary}"

        processed_tokens = self.count_tokens(processed_prompt)
        logger.info(f"Summarised overflow: original {original_tokens} tokens, kept {keep_tokens} tokens, summary added -> new total {processed_tokens} tokens")

        return processed_prompt, {
            "truncated": True,
            "original_token_count": original_tokens,
            "truncated_token_count": processed_tokens,
            "strategy": "summarize_overflow",
            "warning": "Prompt exceeded token limit. The overflow part was summarised and prepended to the kept content."
        }

    async def prepare_prompt(self, prompt: str) -> tuple[str, dict]:
        token_count = self.count_tokens(prompt)
        if token_count <= settings.max_prompt_tokens:
            return prompt, {"truncated": False, "original_token_count": token_count}

        strategy = settings.overflow_strategy
        if strategy == "reject":
            logger.warning(f"Token limit exceeded: {token_count} > {settings.max_prompt_tokens} – rejecting")
            raise TokenLimitExceededError(token_count, settings.max_prompt_tokens)
        elif strategy == "truncate_with_warning":
            return self._truncate_sliding_window(prompt)
        elif strategy == "summarize_overflow":
            return await self._summarize_overflow(prompt)
        else:
            logger.warning(f"Unknown overflow strategy '{strategy}', falling back to 'reject'")
            raise TokenLimitExceededError(token_count, settings.max_prompt_tokens)