"""Build an `LLMBackend` from configuration.

Provider SDKs are imported lazily, so installing only the one you use is enough.
"""
from functools import lru_cache
from typing import Any

from loguru import logger

from src.config import settings
from src.llm.base import LLMBackend, LLMConfigError

_SUPPORTED = ("anthropic", "openai")


@lru_cache(maxsize=None)
def _client(provider: str) -> Any:  # anti-slop: allow no-any-returns - the provider SDKs are imported lazily, so their union cannot be spelled here
    """One shared client per provider; backends differ only by model."""
    if provider == "anthropic":
        import anthropic

        if not settings.ANTHROPIC_API_KEY:
            raise LLMConfigError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    if provider == "openai":
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise LLMConfigError(
                "LLM_PROVIDER=openai requires the openai package "
                "(uv sync --extra openai)"
            ) from exc

        if not settings.OPENAI_API_KEY:
            raise LLMConfigError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        return openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    raise LLMConfigError(f"unknown LLM_PROVIDER {provider!r}; expected one of {_SUPPORTED}")


def build_backend(model: str, provider: str | None = None) -> LLMBackend:
    """Return a backend for `model`, defaulting to the configured provider."""
    provider = (provider or settings.LLM_PROVIDER).strip().lower()
    client = _client(provider)

    if provider == "anthropic":
        from src.llm.anthropic_backend import AnthropicBackend

        backend: LLMBackend = AnthropicBackend(client, model)
    else:
        from src.llm.openai_backend import OpenAIBackend

        backend = OpenAIBackend(client, model)

    logger.info("LLM backend: provider={} model={}", provider, model)
    return backend
