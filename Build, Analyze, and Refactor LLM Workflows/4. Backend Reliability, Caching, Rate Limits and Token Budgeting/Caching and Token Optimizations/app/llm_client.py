# app/llm_client.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import httpx
from loguru import logger
from app.config import settings


class LLMClient:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            timeout=settings.request_timeout_seconds,
        )
        logger.info(f"LLMClient initialized with model: {settings.groq_model}")

    @retry(
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(multiplier=settings.retry_backoff_factor, min=1, max=30),
        retry=retry_if_exception_type(
            (httpx.NetworkError, httpx.TimeoutException, httpx.HTTPStatusError)
        ),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate(self, prompt: str) -> str:
        """Generate a response. No token validation – caller responsible."""
        return await self._call_llm_with_retry(prompt)