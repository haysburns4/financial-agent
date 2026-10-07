"""Build an `LLMBackend` from configuration.

Callers ask for a task's backend (`backend_for("chat")`); which provider and
model serve it is decided here, in `resolve_task`, and nowhere else. Provider
SDKs are imported lazily, so installing only the ones you use is enough.
"""
from functools import lru_cache
from typing import Any, Literal

from loguru import logger

from src.config import Settings, settings
from src.llm.base import LLMBackend, LLMConfigError

Task = Literal["chat", "synthesizer"]

_SUPPORTED = ("anthropic", "openai", "local")


@lru_cache(maxsize=None)
def _client(provider: str) -> Any:  # anti-slop: allow no-any-returns - the provider SDKs are imported lazily, so their union cannot be spelled here
    """One shared client per provider; backends differ only by model."""
    if provider == "anthropic":
        import anthropic

        if not settings.ANTHROPIC_API_KEY:
            raise LLMConfigError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    if provider in ("openai", "local"):
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise LLMConfigError(
                f"LLM_PROVIDER={provider} requires the openai package "
                "(uv sync --extra openai)"
            ) from exc

        if provider == "local":
            # An OpenAI-compatible MLX server: no key, long timeout.
            from src.llm.local_backend import make_client

            return make_client(settings.LOCAL_LLM_BASE_URL, settings.LOCAL_LLM_TIMEOUT_SECONDS)
        if not settings.OPENAI_API_KEY:
            raise LLMConfigError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        return openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    raise LLMConfigError(f"unknown LLM_PROVIDER {provider!r}; expected one of {_SUPPORTED}")


def resolve_task(task: Task, config: Settings = settings) -> tuple[str, str]:
    """(provider, model) for a task.

    The provider is the task's override (LLM_PROVIDER_CHAT /
    LLM_PROVIDER_SYNTHESIZER) or else LLM_PROVIDER. The model is the task's own
    (LLM_CHAT_MODEL / LLM_SYNTHESIS_MODEL) for hosted providers; a local MLX
    server serves one model, LOCAL_LLM_MODEL, whichever task it is.
    """
    match task:
        case "chat":
            override, hosted_model = config.LLM_PROVIDER_CHAT, config.LLM_CHAT_MODEL
        case "synthesizer":
            override, hosted_model = config.LLM_PROVIDER_SYNTHESIZER, config.LLM_SYNTHESIS_MODEL
        case _:
            raise LLMConfigError(f"unknown LLM task {task!r}; expected 'chat' or 'synthesizer'")
    provider = (override or config.LLM_PROVIDER).strip().lower()
    return provider, config.LOCAL_LLM_MODEL if provider == "local" else hosted_model


def backend_for(task: Task) -> LLMBackend:
    """The backend for a task, as configured. Call sites never name a provider."""
    provider, model = resolve_task(task)
    logger.info("LLM task {}: provider={} model={}", task, provider, model)
    return build_backend(model, provider)


def build_backend(model: str, provider: str | None = None) -> LLMBackend:
    """Return a backend for `model`, defaulting to the configured provider."""
    provider = (provider or settings.LLM_PROVIDER).strip().lower()
    client = _client(provider)

    if provider == "anthropic":
        from src.llm.anthropic_backend import AnthropicBackend

        backend: LLMBackend = AnthropicBackend(client, model)
    elif provider == "local":
        from src.llm.local_backend import LocalBackend

        backend = LocalBackend(
            client, model, settings.LOCAL_LLM_BASE_URL, settings.LOCAL_LLM_TIMEOUT_SECONDS
        )
    else:
        from src.llm.openai_backend import OpenAIBackend

        backend = OpenAIBackend(client, model)

    logger.info("LLM backend: provider={} model={}", provider, model)
    return backend
