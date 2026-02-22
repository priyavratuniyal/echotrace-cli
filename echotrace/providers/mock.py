"""
MockProvider — simulated TTFT for when no real LLM backend is available.

Sleeps 800–2000ms to mimic network + inference latency.
"""

from __future__ import annotations

import asyncio
import random

from loguru import logger

from echotrace.analyzers.llm_probe import LLMProbeResult
from echotrace.providers.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Simulated TTFT — always available, used as final fallback."""

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "simulated"

    def is_available(self) -> bool:
        return True

    async def probe(self, transcript: str) -> LLMProbeResult:
        simulated_ttft_sec = random.uniform(0.8, 2.0)
        await asyncio.sleep(simulated_ttft_sec)

        ttft_ms = round(simulated_ttft_sec * 1000, 2)
        fake = f"[MOCK] I can help you with '{transcript[:40]}...'"

        logger.info(f"Mock TTFT: {ttft_ms}ms")

        return LLMProbeResult(
            ttft_ms=ttft_ms,
            response_preview=fake,
            used_mock=True,
        )
