"""Provider-neutral LLM layer.

`src.agent` depends on `LLMBackend` and the neutral types only; concrete
providers live in `anthropic_backend` / `openai_backend` and are selected by
`LLM_PROVIDER`.
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
from src.llm.factory import build_backend

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
    "build_backend",
    "collect",
]
