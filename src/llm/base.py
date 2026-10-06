"""Provider-neutral LLM types and the backend Protocol.

Nothing here imports a vendor SDK. `src.agent` depends only on these types, so
swapping providers is a config change rather than a rewrite.

The surface is streaming-first: `stream()` is the only method a backend must
implement, and `collect()` drains one for callers that just want the finished
message. Streaming is what the AG-UI/CopilotKit frontend needs, so there is no
separate non-streaming path to keep in sync.
"""
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]

# Provider stop reasons are mapped onto these; "other" is the catch-all so a
# new vendor value never crashes a caller.
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "other"]


class LLMError(Exception):
    """Base for every backend failure, so callers never catch a vendor type."""


class LLMConfigError(LLMError):
    """Provider is unknown, uninstalled, or missing credentials."""


class LLMConnectionError(LLMError):
    """Network-level failure; retryable."""


class LLMRateLimitError(LLMError):
    """Provider rate limit; retryable after `retry_after` seconds when known."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMStatusError(LLMError):
    """Non-retryable API error (bad request, auth, not found)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ToolCall:
    """A model request to run a tool. `arguments` is already JSON-parsed."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """One conversation turn.

    A turn carrying `tool_results` answers a previous assistant `tool_calls`
    turn. Providers encode that differently (Anthropic: tool_result blocks in a
    user message; OpenAI: one `tool` message each), so backends translate.
    """

    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class ToolDef:
    """A tool offered to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of a tool call, for UIs that show one being composed.

    `arguments_json` is not valid JSON on its own — run tools from the parsed
    `tool_calls` on `MessageComplete`.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments_json: str = ""


@dataclass(frozen=True)
class MessageComplete:
    """Terminal event. Every backend must yield exactly one, last."""

    message: Message
    stop_reason: StopReason
    usage: Usage


Delta = TextDelta | ToolCallDelta | MessageComplete


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = "end_turn"
    usage: Usage = field(default_factory=Usage)


class LLMBackend(Protocol):
    """What `src.agent` depends on. Implemented per provider in this package."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] = (),
        max_tokens: int = 4096,
        **extra: Any,  # anti-slop: allow no-any-parameters - provider-specific knobs are opaque here by design
    ) -> AsyncIterator[Delta]:
        """Stream a completion.

        `extra` carries provider-specific knobs and is ignored by backends that
        do not understand a key, so passing one never breaks portability.

        Raises `LLMError` subclasses only — vendor exceptions are translated.
        """
        ...


async def collect(stream: AsyncIterator[Delta]) -> LLMResponse:
    """Drain a stream into its finished response."""
    async for delta in stream:
        if isinstance(delta, MessageComplete):
            return LLMResponse(
                text=delta.message.text,
                tool_calls=delta.message.tool_calls,
                stop_reason=delta.stop_reason,
                usage=delta.usage,
            )
    raise LLMError("stream ended without a terminal MessageComplete")
