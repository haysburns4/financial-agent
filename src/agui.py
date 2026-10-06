"""AG-UI protocol adapter for CopilotKit.

Translates `AgentChat.ask_stream` deltas into the AG-UI event stream that
CopilotKit consumes over SSE. The agent knows nothing about this protocol and
this module knows nothing about a provider — both meet at the `src.llm` types.
"""
import uuid
from collections.abc import AsyncIterator, Sequence

from ag_ui.core import (
    AssistantMessage,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    UserMessage,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from src.agent.chat import AgentChat
from src.llm import MessageComplete, TextDelta, ToolCallDelta, ToolResult


def create_agui_router(chat: AgentChat) -> APIRouter:
    router = APIRouter()

    @router.post("/agui")
    async def agui(body: RunAgentInput, request: Request) -> StreamingResponse:
        encoder = EventEncoder(accept=request.headers.get("accept"))
        return StreamingResponse(
            _run(chat, body, encoder),
            media_type=encoder.get_content_type(),
        )

    return router


def split_conversation(messages: Sequence) -> tuple[str, list[dict]]:
    """Split an AG-UI transcript into (question, history) for `AgentChat`.

    The last user message is the question; user and assistant turns before it
    become history. System turns are dropped — `AgentChat` builds its own system
    prompt — as are tool turns, since `AgentChat` re-runs its own tools rather
    than trusting replayed results.
    """
    last_user = max(
        (i for i, m in enumerate(messages) if isinstance(m, UserMessage)),
        default=-1,
    )
    if last_user < 0:
        raise ValueError("conversation contains no user message")

    history = [
        {"role": "user" if isinstance(m, UserMessage) else "assistant", "content": m.content}
        for m in messages[:last_user]
        if isinstance(m, UserMessage | AssistantMessage) and m.content
    ]
    return messages[last_user].content, history


async def _run(
    chat: AgentChat,
    body: RunAgentInput,
    encoder: EventEncoder,
) -> AsyncIterator[str]:
    yield encoder.encode(RunStartedEvent(thread_id=body.thread_id, run_id=body.run_id))

    if body.tools:
        logger.warning(
            "agui: ignoring {} frontend tool(s); only the agent's own tools are offered",
            len(body.tools),
        )

    # One AG-UI run spans several model turns (model -> tools -> model). Each
    # turn's text gets a fresh message id, and its tool calls are ended when the
    # turn completes so their results can follow before the next turn starts.
    message_id = uuid.uuid4().hex
    text_open = False
    tool_ids: dict[int, str] = {}
    final: MessageComplete | None = None

    try:
        question, history = split_conversation(body.messages)

        async for delta in chat.ask_stream(question, history):
            if isinstance(delta, TextDelta):
                if not text_open:
                    yield encoder.encode(
                        TextMessageStartEvent(message_id=message_id, role="assistant")
                    )
                    text_open = True
                yield encoder.encode(
                    TextMessageContentEvent(message_id=message_id, delta=delta.text)
                )

            elif isinstance(delta, ToolCallDelta):
                if text_open:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                    text_open = False
                if delta.id and delta.name:
                    tool_ids[delta.index] = delta.id
                    yield encoder.encode(
                        ToolCallStartEvent(
                            tool_call_id=delta.id,
                            tool_call_name=delta.name,
                            parent_message_id=message_id,
                        )
                    )
                elif delta.arguments_json and (call_id := tool_ids.get(delta.index)):
                    yield encoder.encode(
                        ToolCallArgsEvent(tool_call_id=call_id, delta=delta.arguments_json)
                    )

            elif isinstance(delta, MessageComplete):
                final = delta
                if text_open:
                    yield encoder.encode(TextMessageEndEvent(message_id=message_id))
                    text_open = False
                for call_id in tool_ids.values():
                    yield encoder.encode(ToolCallEndEvent(tool_call_id=call_id))
                tool_ids = {}
                message_id = uuid.uuid4().hex

            elif isinstance(delta, ToolResult):
                yield encoder.encode(
                    ToolCallResultEvent(
                        message_id=uuid.uuid4().hex,
                        tool_call_id=delta.call_id,
                        content=delta.content,
                        role="tool",
                    )
                )

    except Exception as exc:
        logger.exception("agui: run {} failed", body.run_id)
        if text_open:
            yield encoder.encode(TextMessageEndEvent(message_id=message_id))
        yield encoder.encode(RunErrorEvent(message=str(exc)))
        return

    yield encoder.encode(
        RunFinishedEvent(
            thread_id=body.thread_id,
            run_id=body.run_id,
            result=_result(final),
        )
    )


def _result(final: MessageComplete | None) -> dict | None:
    if final is None:
        return None
    return {
        "stop_reason": final.stop_reason,
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }
