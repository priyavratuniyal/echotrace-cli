"""
Task C — LLM Probe (TTFT Measurement).

Delegates to the active LLM provider to measure Time-to-First-Token.
The provider is injected via dependency injection from the Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from echotrace.core.telemetry import TelemetryCollector


@dataclass
class LLMProbeResult:
    ttft_ms: float
    response_preview: str
    used_mock: bool
    provider_name: str = ""
    provider_model: str = ""


class LLMProbe:
    """
    Thin adapter around BaseLLMProvider.

    Handles empty-prompt edge case and error fallback,
    then delegates to the active provider's probe() method.
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        provider: Optional["BaseLLMProvider"] = None,
    ) -> None:
        self._collector = collector
        self._provider = provider

    async def probe(self, prompt_text: str) -> LLMProbeResult:
        """Run the TTFT probe via the configured provider."""
        if not prompt_text.strip():
            pname = self._provider.provider_name if self._provider else "mock"
            pmodel = self._provider.model_name if self._provider else ""
            return LLMProbeResult(
                ttft_ms=0.0,
                response_preview="",
                used_mock=True,
                provider_name=pname,
                provider_model=pmodel,
            )

        # Fallback to mock if no provider was injected
        if self._provider is None:
            from echotrace.providers.mock import MockProvider

            self._provider = MockProvider()

        try:
            result = await self._provider.probe(prompt_text)
            result.provider_name = self._provider.provider_name
            result.provider_model = self._provider.model_name
            return result
        except Exception as e:
            logger.warning(
                f"{self._provider.provider_name} probe failed, "
                f"falling back to mock: {e}"
            )
            from echotrace.providers.mock import MockProvider

            mock = MockProvider()
            result = await mock.probe(prompt_text)
            result.provider_name = mock.provider_name
            result.provider_model = mock.model_name
            return result
