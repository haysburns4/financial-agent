"""OpenAI implementation of `LLMBackend`.

Exists to keep `src.llm.base` honest: an abstraction with one implementation
tends to quietly leak that provider's shape. The `_to_*` helpers are pure and
unit-tested; the streaming call itself has not been run against a live OpenAI
endpoint from this repo.

`openai` is an optional dependency — `uv sync --extra openai`.
"""
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai

from src.llm.base import (
    Delta,
    LLMConnectionError,
    LLMRateLimitError,
    LLMStatusError,
    Message,
    MessageComplete,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolDef,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _to_openai_tools(tools: Sequence[ToolDef]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: Sequence[Message]) -> list[dict]:
    """Neutral turns -> Chat Completions turns.

    Unlike Anthropic, the system prompt is a message rather than a top-level
    field, and each tool result is its own `tool` message.
    """
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.tool_results:
            out.extend(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                for r in m.tool_results
            )
        elif m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.text or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in m.tool_calls
                    ],
                }
            )
        elif m.text:
            out.append({"role": m.role, "content": m.text})
    return out


class OpenAIBackend:
    provider = "openai"

    def __init__(self, client: openai.AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] = (),
        max_tokens: int = 4096,
        **extra: Any,
    ) -> AsyncIterator[Delta]:
        request: dict[str, Any] = {
            "model": self._model,
            # `max_tokens` is deprecated and rejected by reasoning models.
            "max_completion_tokens": max_tokens,
            "messages": _to_openai_messages(system, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **({"tools": _to_openai_tools(tools)} if tools else {}),
            **extra,
        }

        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}  # tool calls arrive keyed by index, not id
        stop_reason: StopReason = "end_turn"
        usage = Usage()

        try:
            async for chunk in await self._client.chat.completions.create(**request):
                if chunk.usage is not None:
                    usage = _to_usage(chunk.usage)
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    stop_reason = _STOP_REASONS.get(choice.finish_reason, "other")
                if choice.delta is None:
                    continue

                if choice.delta.content:
                    text_parts.append(choice.delta.content)
                    yield TextDelta(choice.delta.content)

                for call in choice.delta.tool_calls or ():
                    fn = call.function
                    slot = calls.setdefault(call.index, {"id": "", "name": "", "args": ""})
                    slot["id"] = call.id or slot["id"]
                    slot["name"] = (fn.name if fn else "") or slot["name"]
                    fragment = (fn.arguments if fn else "") or ""
                    slot["args"] += fragment
                    yield ToolCallDelta(
                        index=call.index,
                        id=call.id,
                        name=fn.name if fn else None,
                        arguments_json=fragment,
                    )
        except openai.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            raise LLMRateLimitError(
                str(exc), int(retry_after) if retry_after else None
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise LLMStatusError(str(exc), exc.status_code) from exc

        yield MessageComplete(
            message=Message(
                role="assistant",
                text="".join(text_parts),
                tool_calls=tuple(
                    ToolCall(id=s["id"], name=s["name"], arguments=_safe_json(s["args"]))
                    for _, s in sorted(calls.items())
                ),
            ),
            stop_reason=stop_reason,
            usage=usage,
        )


def _to_usage(raw: Any) -> Usage:
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        input_tokens=raw.prompt_tokens or 0,
        output_tokens=raw.completion_tokens or 0,
        cache_read_tokens=(details.cached_tokens if details else 0) or 0,
    )


def _safe_json(raw: str) -> dict[str, Any]:
    """Parse accumulated argument fragments, tolerating a truncated stream."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
