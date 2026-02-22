"""
Provider resolver — auto-detects the best available LLM backend.

Priority: Groq → Ollama → HuggingFace → Mock.
Can be overridden via config.llm.provider.
"""

from __future__ import annotations

from loguru import logger

from echotrace.config import EchoTraceConfig
from echotrace.providers.base import BaseLLMProvider


def resolve_provider(config: EchoTraceConfig) -> BaseLLMProvider:
    """
    Resolve and return the appropriate LLM provider.

    If config.llm.provider is "auto", tries providers in priority order.
    Otherwise uses the explicitly requested provider.
    """
    provider_name = config.llm.provider.lower().strip()

    if provider_name != "auto":
        return _get_explicit_provider(provider_name, config)

    return _auto_detect(config)


def _get_explicit_provider(
    name: str, config: EchoTraceConfig
) -> BaseLLMProvider:
    """Instantiate a specific provider by name."""
    if name == "groq":
        from echotrace.providers.groq import GroqProvider

        return GroqProvider(
            api_key=config.llm.groq_api_key,
            model=config.llm.groq_model,
        )

    if name == "ollama":
        from echotrace.providers.ollama import OllamaProvider

        return OllamaProvider(
            model=config.llm.ollama_model,
            host=config.llm.ollama_host,
        )

    if name == "huggingface":
        from echotrace.providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider(
            api_key=config.llm.hf_api_key,
            model=config.llm.hf_model,
        )

    if name == "mock":
        from echotrace.providers.mock import MockProvider

        return MockProvider()

    logger.warning(f"Unknown provider '{name}', falling back to mock")
    from echotrace.providers.mock import MockProvider

    return MockProvider()


def _auto_detect(config: EchoTraceConfig) -> BaseLLMProvider:
    """Try providers in priority order, return first available."""

    # 1. Groq (fastest cloud inference)
    if config.llm.groq_api_key:
        from echotrace.providers.groq import GroqProvider

        provider = GroqProvider(
            api_key=config.llm.groq_api_key,
            model=config.llm.groq_model,
        )
        if provider.is_available():
            logger.info(f"Auto-detected provider: {provider.label()}")
            return provider

    # 2. Ollama (local, zero cost)
    try:
        from echotrace.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            model=config.llm.ollama_model,
            host=config.llm.ollama_host,
        )
        if provider.is_available():
            logger.info(f"Auto-detected provider: {provider.label()}")
            return provider
    except ImportError:
        pass

    # 3. HuggingFace Inference API
    if config.llm.hf_api_key:
        try:
            from echotrace.providers.huggingface import HuggingFaceProvider

            provider = HuggingFaceProvider(
                api_key=config.llm.hf_api_key,
                model=config.llm.hf_model,
            )
            if provider.is_available():
                logger.info(f"Auto-detected provider: {provider.label()}")
                return provider
        except ImportError:
            pass

    # 4. Mock fallback
    from echotrace.providers.mock import MockProvider

    logger.info("No LLM provider available — using mock fallback")
    return MockProvider()
