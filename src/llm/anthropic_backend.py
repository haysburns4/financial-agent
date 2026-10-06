"""Anthropic implementation of `LLMBackend`.

Wire translation lives in the module-level `_to_*` helpers so it can be tested
without a client, a key, or a network call.
"""
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic.lib.streaming import ParsedMessageStreamEvent

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
    ToolResult,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def _to_anthropic_tools(tools: Sequence[ToolDef]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _tool_result_block(result: ToolResult) -> dict:
    block = {
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": result.content,
    }
    if result.is_error:
        block["is_error"] = True
    return block


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict]:
    """Neutral turns -> Messages API turns."""
    out: list[dict] = []
    for m in messages:
        if m.tool_results:
            # The API rejects tool_result blocks split across several messages.
            out.append(
                {
                    "role": "user",
                    "content": [_tool_result_block(r) for r in m.tool_results],
                }
            )
            continue

        blocks: list[dict] = []
        if m.text:
            blocks.append({"type": "text", "text": m.text})
        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in m.tool_calls
        )
        if blocks:
            out.append({"role": m.role, "content": blocks})
    return out


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, client: anthropic.AsyncAnthropic, model: str) -> None:
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
        **extra: Any,  # anti-slop: allow no-any-parameters - provider passthrough is the documented Protocol contract
    ) -> AsyncIterator[Delta]:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _to_anthropic_messages(messages),
        }
        if tools:
            request["tools"] = _to_anthropic_tools(tools)
        # Opus 5 runs adaptive thinking when `thinking` is omitted, so send
        # nothing by default rather than pinning a setting for the caller.
        request.update(extra)

        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    if (delta := _event_to_delta(event)) is not None:
                        yield delta
                final = await stream.get_final_message()
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after")
            raise LLMRateLimitError(
                str(exc), int(retry_after) if retry_after else None
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise LLMStatusError(str(exc), exc.status_code) from exc

        yield MessageComplete(
            message=Message(
                role="assistant",
                text="".join(b.text for b in final.content if b.type == "text"),
                tool_calls=tuple(
                    ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
                    for b in final.content
                    if b.type == "tool_use"
                ),
            ),
            stop_reason=_STOP_REASONS.get(final.stop_reason or "", "other"),
            usage=Usage(
                input_tokens=final.usage.input_tokens or 0,
                output_tokens=final.usage.output_tokens or 0,
                cache_read_tokens=final.usage.cache_read_input_tokens or 0,
            ),
        )


def _event_to_delta(event: ParsedMessageStreamEvent) -> Delta | None:
    """One SDK stream event -> a neutral delta, or None to ignore it."""
    if event.type == "content_block_start" and event.content_block.type == "tool_use":
        return ToolCallDelta(
            index=event.index,
            id=event.content_block.id,
            name=event.content_block.name,
        )
    if event.type != "content_block_delta":
        return None
    if event.delta.type == "text_delta":
        return TextDelta(event.delta.text)
    if event.delta.type == "input_json_delta":
        return ToolCallDelta(index=event.index, arguments_json=event.delta.partial_json)
    return None
