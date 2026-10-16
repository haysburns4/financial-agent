"""Provider-neutral LLM layer.

`src.agent` depends on `LLMBackend` and the neutral types only; concrete
providers live in `anthropic_backend` / `openai_backend` / `local_backend`
(any OpenAI-compatible server on this machine or the LAN) and are selected
by `LLM_PROVIDER`.

The factory is imported on first use rather than here: it reads src.config,
which raises without E-Trade keys, and setup tooling (src/cli) must be able
to import `src.llm.local_profiles` before those exist.
"""
from typing import TYPE_CHECKING

from src.llm.base import (
    Delta,
    LLMBackend,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMStatusError,
    Message,
    MessageComplete,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolDef,
    ToolResult,
    Usage,
    collect,
    estimate_tokens,
)

if TYPE_CHECKING:
    from src.llm.factory import backend_for, build_backend, resolve_task

_FACTORY_NAMES = frozenset({"backend_for", "build_backend", "resolve_task"})


def __getattr__(name: str) -> object:
    if name in _FACTORY_NAMES:
        from src.llm import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Delta",
    "LLMBackend",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMStatusError",
    "Message",
    "MessageComplete",
    "TextDelta",
    "ToolCall",
    "ToolCallDelta",
    "ToolDef",
    "ToolResult",
    "Usage",
    "backend_for",
    "build_backend",
    "collect",
    "estimate_tokens",
    "resolve_task",
]
