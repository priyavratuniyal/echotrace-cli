"""
OllamaProvider — measures TTFT via a locally-running Ollama server.

Uses the ollama Python SDK for streaming. Cold starts (model not loaded
in memory) can take 3–8s which inflates TTFT — this is expected and
documented in the probe result.
"""

from __future__ import annotations

import time

from loguru import logger

from echotrace.analyzers.llm_probe import LLMProbeResult
from echotrace.providers.base import BaseLLMProvider, SYSTEM_PROMPT


class OllamaProvider(BaseLLMProvider):
    """Local Ollama server — zero API cost, works offline."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._host = host

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Ping Ollama's /api/tags endpoint to check if it's running."""
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                f"{self._host}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2):
                return True
        except Exception:
            return False

    async def probe(self, transcript: str) -> LLMProbeResult:
        try:
            import ollama as ollama_sdk
        except ImportError:
            raise ImportError(
                "ollama package not installed. "
                "Install with: pip install echotrace[ollama]"
            )

        client = ollama_sdk.AsyncClient(host=self._host)

        start = time.perf_counter()
        first_token_time = None
        full_response = ""

        stream = await client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            stream=True,
            options={"num_predict": 50},
        )

        async for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_response += content

        if first_token_time is None:
            first_token_time = time.perf_counter()

        ttft_ms = round((first_token_time - start) * 1000, 2)
        logger.info(
            f"Ollama TTFT: {ttft_ms}ms (model={self._model}, "
            f"host={self._host})"
        )

        return LLMProbeResult(
            ttft_ms=ttft_ms,
            response_preview=full_response[:100].strip(),
            used_mock=False,
        )
