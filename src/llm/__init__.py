"""Provider-neutral LLM layer.

`src.agent` depends on `LLMBackend` and the neutral types only; concrete
providers live in `anthropic_backend` / `openai_backend` / `local_backend`
(an OpenAI-compatible MLX server) and are selected by `LLM_PROVIDER`.
"""
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
)
from src.llm.factory import backend_for, build_backend, resolve_task

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
    "resolve_task",
]
