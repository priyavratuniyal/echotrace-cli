"""
HuggingFaceProvider — measures TTFT via HuggingFace Inference API.

Uses the cloud Inference API (not local transformers) for lightweight,
no-GPU-required probing. No streaming — measures full round-trip as TTFT.
"""

from __future__ import annotations

import time

from loguru import logger

from echotrace.analyzers.llm_probe import LLMProbeResult
from echotrace.providers.base import BaseLLMProvider, SYSTEM_PROMPT


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API — cloud-based, free tier available."""

    def __init__(
        self,
        api_key: str,
        model: str = "microsoft/Phi-3-mini-4k-instruct",
    ) -> None:
        self._api_key = api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def probe(self, transcript: str) -> LLMProbeResult:
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface-hub package not installed. "
                "Install with: pip install echotrace[hf]"
            )

        import asyncio

        client = InferenceClient(token=self._api_key)
        prompt = f"System: {SYSTEM_PROMPT}\nUser: {transcript}\nAssistant:"

        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        # HF Inference API — no streaming, measure full round-trip
        response = await loop.run_in_executor(
            None,
            lambda: client.text_generation(
                prompt,
                model=self._model,
                max_new_tokens=50,
            ),
        )

        ttft_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"HuggingFace TTFT: {ttft_ms}ms (model={self._model})"
        )

        return LLMProbeResult(
            ttft_ms=ttft_ms,
            response_preview=str(response)[:100].strip(),
            used_mock=False,
        )
