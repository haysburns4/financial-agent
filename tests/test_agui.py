"""The AG-UI adapter: transcript in, protocol event stream out."""
import json
from datetime import datetime, timezone

import httpx
import pytest
from ag_ui.core import AssistantMessage, Context, SystemMessage, UserMessage
from fastapi import FastAPI
from sqlalchemy import insert

from src.agent.chat import AgentChat
from src.agui import create_agui_router, highlighted_account, split_conversation
from src.llm import LLMStatusError
from src.models import Position

from tests.conftest import FakeBackend, text_reply, tool_reply


class BoomBackend(FakeBackend):
    async def stream(self, **kwargs):
        raise LLMStatusError("upstream exploded", 500)
        yield  # unreachable; makes this an async generator


def _user(text: str, mid: str = "u1") -> dict:
    return {"id": mid, "role": "user", "content": text}


async def _post(engine, backend, messages: list[dict], **extra) -> list[dict]:
    """Run one AG-UI request and return the decoded SSE events."""
    app = FastAPI()
    app.include_router(create_agui_router(AgentChat(engine, backend)))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.post(
            "/agui",
            json={"thread_id": "t1", "run_id": "r1", "messages": messages, **extra},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def test_text_run_emits_the_full_lifecycle(engine, backend):
    events = await _post(engine, backend, [_user("What do I hold?")])

    assert [e["type"] for e in events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert events[0]["threadId"] == "t1"
    assert events[0]["runId"] == "r1"
    assert events[2]["delta"] == "AAPL is your largest position."
    # Every text event belongs to one message.
    ids = {e["messageId"] for e in events if e["type"].startswith("TEXT_MESSAGE")}
    assert len(ids) == 1


async def test_run_finished_reports_stop_reason_and_usage(engine, backend):
    events = await _post(engine, backend, [_user("hi")])
    assert events[-1]["result"] == {
        "stop_reason": "end_turn",
        "input_tokens": 10,
        "output_tokens": 5,
    }


async def test_last_user_message_is_the_question(engine, backend):
    await _post(
        engine,
        backend,
        [
            _user("How is AAPL?", "u1"),
            {"id": "a1", "role": "assistant", "content": "Flat."},
            _user("And MSFT?", "u2"),
        ],
    )

    sent = [(m.role, m.text) for m in backend.calls[0]["messages"]]
    assert sent == [("user", "How is AAPL?"), ("assistant", "Flat."), ("user", "And MSFT?")]


async def test_missing_user_message_is_a_run_error(engine, backend):
    events = await _post(engine, backend, [{"id": "a1", "role": "assistant", "content": "hi"}])

    types = [e["type"] for e in events]
    # The run is already committed once RUN_STARTED is on the wire.
    assert types == ["RUN_STARTED", "RUN_ERROR"]
    assert "no user message" in events[-1]["message"]


async def test_backend_failure_is_a_run_error(engine):
    events = await _post(engine, BoomBackend(), [_user("hi")])

    assert [e["type"] for e in events] == ["RUN_STARTED", "RUN_ERROR"]
    assert "upstream exploded" in events[-1]["message"]


async def test_a_tool_run_streams_call_then_result_then_answer(engine):
    backend = FakeBackend(
        [tool_reply("get_positions", {"account_id": "all"}), text_reply("You hold nothing.")]
    )
    events = await _post(engine, backend, [_user("what do I hold")])

    assert [e["type"] for e in events] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    assert events[1]["toolCallName"] == "get_positions"
    assert events[2]["delta"] == '{"account_id": "all"}'
    # Start, args, end and result all address the same call.
    assert {e["toolCallId"] for e in events[1:5]} == {"c1"}
    assert events[4]["content"] == "[]"  # empty DB, but the tool really ran


async def test_text_before_a_tool_call_is_closed_and_reopened(engine):
    # Text and tool calls interleave across turns; each text block needs its own
    # id, or the frontend appends turn 2 onto a message it already closed.
    backend = FakeBackend(
        [
            [*text_reply("Let me check.")[:1], *tool_reply("get_positions", {})],
            text_reply("Nothing."),
        ]
    )
    events = await _post(engine, backend, [_user("holdings")])

    text_ids = [e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_START"]
    assert len(text_ids) == 2
    assert len(set(text_ids)) == 2


async def test_frontend_tools_are_accepted_but_not_forwarded(engine, backend):
    # CopilotKit sends its frontend actions on every request; AgentChat has no
    # tool surface yet, so the run must still succeed rather than 422.
    events = await _post(
        engine,
        backend,
        [_user("hi")],
        tools=[{"name": "highlight", "description": "highlight a ticker", "parameters": {}}],
    )
    assert events[-1]["type"] == "RUN_FINISHED"
    # The model is offered the agent's own tools, never the frontend's.
    assert "highlight" not in {t.name for t in backend.calls[0]["tools"]}


def test_split_conversation_drops_system_and_keeps_order():
    question, history = split_conversation(
        [
            SystemMessage(id="s1", content="ignored"),
            UserMessage(id="u1", content="first"),
            AssistantMessage(id="a1", content="reply"),
            UserMessage(id="u2", content="second"),
        ]
    )
    assert question == "second"
    assert history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]


def test_split_conversation_rejects_a_transcript_with_no_user_turn():
    with pytest.raises(ValueError, match="no user message"):
        split_conversation([AssistantMessage(id="a1", content="hi")])


# ---------- the highlighted account ----------


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ([{"description": "highlighted_account", "value": "84719991"}], "84719991"),
        ([{"description": "highlighted_account", "value": "all"}], None),
        ([{"description": "highlighted_account", "value": " "}], None),
        ([{"description": "something else", "value": "84719991"}], None),
        ([], None),
    ],
)
def test_highlighted_account_is_read_from_the_run_context(context, expected):
    assert highlighted_account([Context(**c) for c in context]) == expected


async def test_the_dashboards_highlight_scopes_the_agents_tools(engine):
    async with engine.begin() as conn:
        await conn.execute(insert(Position), [
            {"account_id": acct, "ticker": ticker, "quantity": 1.0, "cost_basis": 100.0,
             "market_value": 100.0, "last_updated": datetime.now(timezone.utc)}
            for acct, ticker in [("84719991", "AAPL"), ("55520002", "MSFT")]
        ])
    backend = FakeBackend([tool_reply("get_positions", {}), text_reply("You hold MSFT.")])

    events = await _post(
        engine, backend, [_user("what do I hold")],
        context=[{"description": "highlighted_account", "value": "55520002"}],
    )

    result = next(e for e in events if e["type"] == "TOOL_CALL_RESULT")
    assert [p["ticker"] for p in json.loads(result["content"])] == ["MSFT"]
    assert "••0002" in backend.calls[0]["system"]
