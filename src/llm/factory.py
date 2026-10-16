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
def _client(provider: str, credential: str | None, timeout_seconds: float = 0.0) -> Any:  # anti-slop: allow no-any-returns - the provider SDKs are imported lazily, so their union cannot be spelled here
    """One shared client per provider and endpoint; backends differ only by
    model. `credential` is the API key, or the base URL for local, the only
    provider with a timeout of its own."""
    if provider == "anthropic":
        import anthropic

        if not credential:
            raise LLMConfigError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        return anthropic.AsyncAnthropic(api_key=credential)

    if provider in ("openai", "local"):
        try:
            import openai
        except ModuleNotFoundError as exc:
            raise LLMConfigError(
                f"LLM_PROVIDER={provider} requires the openai package "
                "(uv sync --extra openai)"
            ) from exc

        if provider == "local":
            # An OpenAI-compatible server: no key, long timeout.
            from src.llm.local_backend import make_client

            return make_client(credential or "", timeout_seconds)
        if not credential:
            raise LLMConfigError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        return openai.AsyncOpenAI(api_key=credential)

    raise LLMConfigError(f"unknown LLM_PROVIDER {provider!r}; expected one of {_SUPPORTED}")


def resolve_task(task: Task, config: Settings = settings) -> tuple[str, str]:
    """(provider, model) for a task.

    The provider is the task's override (LLM_PROVIDER_CHAT /
    LLM_PROVIDER_SYNTHESIZER) or else LLM_PROVIDER. The model is the task's own
    (LLM_CHAT_MODEL / LLM_SYNTHESIS_MODEL) for hosted providers; for local it
    is LOCAL_LLM_MODEL_CHAT / LOCAL_LLM_MODEL_SYNTHESIZER, else LOCAL_LLM_MODEL.
    """
    match task:
        case "chat":
            override, hosted_model, local_model = (
                config.LLM_PROVIDER_CHAT, config.LLM_CHAT_MODEL, config.LOCAL_LLM_MODEL_CHAT
            )
        case "synthesizer":
            override, hosted_model, local_model = (
                config.LLM_PROVIDER_SYNTHESIZER, config.LLM_SYNTHESIS_MODEL, config.LOCAL_LLM_MODEL_SYNTHESIZER
            )
        case _:
            raise LLMConfigError(f"unknown LLM task {task!r}; expected 'chat' or 'synthesizer'")
    provider = (override or config.LLM_PROVIDER).strip().lower()
    if provider == "local":
        return provider, local_model or config.LOCAL_LLM_MODEL
    return provider, hosted_model


def backend_for(task: Task, config: Settings = settings) -> LLMBackend:
    """The backend for a task, as configured. Call sites never name a provider."""
    provider, model = resolve_task(task, config)
    logger.info("LLM task {}: provider={} model={}", task, provider, model)
    return build_backend(model, provider, config)


def build_backend(model: str, provider: str | None = None, config: Settings = settings) -> LLMBackend:
    """Return a backend for `model`, defaulting to the configured provider."""
    provider = (provider or config.LLM_PROVIDER).strip().lower()

    if provider == "anthropic":
        from src.llm.anthropic_backend import AnthropicBackend

        client = _client(provider, config.ANTHROPIC_API_KEY)
        backend: LLMBackend = AnthropicBackend(client, model, context_tokens=config.LLM_CONTEXT_BUDGET)
    elif provider == "local":
        from src.llm.health import monitor_for
        from src.llm.local_backend import LocalBackend
        from src.llm.local_profiles import model_profile, server_profile

        client = _client(provider, config.LOCAL_LLM_BASE_URL, config.LOCAL_LLM_TIMEOUT_SECONDS)
        server = server_profile(config.LOCAL_LLM_SERVER)
        backend = LocalBackend(
            client, model, config.LOCAL_LLM_BASE_URL, config.LOCAL_LLM_TIMEOUT_SECONDS,
            server=server,
            family=model_profile(model, config.LOCAL_LLM_MODEL_FAMILY),
            context_tokens=config.LOCAL_LLM_CONTEXT_BUDGET,
            monitor=monitor_for(config.LOCAL_LLM_BASE_URL, server.kind),
        )
    else:
        from src.llm.openai_backend import OpenAIBackend

        client = _client(provider, config.OPENAI_API_KEY)
        backend = OpenAIBackend(client, model, context_tokens=config.LLM_CONTEXT_BUDGET)

    logger.info("LLM backend: provider={} model={}", provider, model)
    return backend
