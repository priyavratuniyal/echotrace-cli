"""
BaseLLMProvider — abstract interface for all LLM backends.

Every provider implements probe() → LLMProbeResult and
is_available() for health checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from echotrace.analyzers.llm_probe import LLMProbeResult


SYSTEM_PROMPT = "You are a voice assistant. Answer in one sentence."


class BaseLLMProvider(ABC):
    """Abstract base for LLM TTFT measurement backends."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. 'groq', 'ollama'."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, e.g. 'llama3-8b-8192'."""
        ...

    @abstractmethod
    async def probe(self, transcript: str) -> LLMProbeResult:
        """
        Send transcript to the LLM and measure Time-to-First-Token.

        Returns an LLMProbeResult with ttft_ms, response_preview,
        and used_mock flag.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Synchronous health check — can this provider be used right now?

        For cloud APIs this checks if an API key is configured.
        For Ollama this pings the local server.
        """
        ...

    def label(self) -> str:
        """Display label for the waterfall, e.g. 'groq (llama3-8b-8192)'."""
        return f"{self.provider_name} ({self.model_name})"
