"""
GroqProvider — measures TTFT via Groq's streaming API.

Refactored from the original LLMProbe._probe_groq().
"""

from __future__ import annotations

import time

from loguru import logger

from echotrace.analyzers.llm_probe import LLMProbeResult
from echotrace.providers.base import BaseLLMProvider, SYSTEM_PROMPT


class GroqProvider(BaseLLMProvider):
    """Groq cloud streaming — fast inference, free tier available."""

    def __init__(self, api_key: str, model: str = "llama3-8b-8192") -> None:
        self._api_key = api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def probe(self, transcript: str) -> LLMProbeResult:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=self._api_key)

        start = time.perf_counter()
        first_token_time = None
        full_response = ""

        stream = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            stream=True,
            max_tokens=50,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_response += delta.content

        if first_token_time is None:
            first_token_time = time.perf_counter()

        ttft_ms = round((first_token_time - start) * 1000, 2)
        logger.info(f"Groq TTFT: {ttft_ms}ms (model={self._model})")

        return LLMProbeResult(
            ttft_ms=ttft_ms,
            response_preview=full_response[:100].strip(),
            used_mock=False,
        )
