"""Per-provider wire translation.

These are the parts most likely to drift as the layer grows, and they are pure
functions — no client, no key, no network.
"""
import json

import pytest

from src.llm import Message, ToolCall, ToolDef, ToolResult
from src.llm.anthropic_backend import _to_anthropic_messages, _to_anthropic_tools

TOOL = ToolDef(
    name="get_positions",
    description="Current portfolio positions.",
    parameters={"type": "object", "properties": {}},
)
CALL = ToolCall(id="t1", name="get_positions", arguments={"account_id": "X"})


# ---------- anthropic ----------


def test_anthropic_tools_use_input_schema():
    assert _to_anthropic_tools([TOOL]) == [
        {
            "name": "get_positions",
            "description": "Current portfolio positions.",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_anthropic_text_turns():
    out = _to_anthropic_messages(
        [Message(role="user", text="hi"), Message(role="assistant", text="hello")]
    )
    assert out == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]


def test_anthropic_tool_calls_become_tool_use_blocks():
    out = _to_anthropic_messages([Message(role="assistant", text="checking", tool_calls=(CALL,))])
    assert out[0]["content"] == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "t1", "name": "get_positions", "input": {"account_id": "X"}},
    ]


def test_anthropic_tool_results_share_one_user_message():
    # The API rejects tool_result blocks split across several messages.
    out = _to_anthropic_messages(
        [
            Message(
                role="user",
                tool_results=(
                    ToolResult(call_id="t1", content="ok"),
                    ToolResult(call_id="t2", content="boom", is_error=True),
                ),
            )
        ]
    )
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert out[0]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True},
    ]


def test_anthropic_drops_empty_turns():
    assert _to_anthropic_messages([Message(role="user", text="")]) == []


# ---------- openai ----------

pytest.importorskip("openai", reason="optional provider; uv sync --extra openai")

from src.llm.openai_backend import _to_openai_messages, _to_openai_tools  # noqa: E402


def test_openai_tools_are_function_wrapped():
    assert _to_openai_tools([TOOL]) == [
        {
            "type": "function",
            "function": {
                "name": "get_positions",
                "description": "Current portfolio positions.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_openai_system_becomes_the_first_message():
    out = _to_openai_messages("be brief", [Message(role="user", text="hi")])
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_openai_tool_calls_serialise_arguments_as_json():
    out = _to_openai_messages("s", [Message(role="assistant", tool_calls=(CALL,))])
    call = out[1]["tool_calls"][0]
    assert out[1]["content"] is None
    assert call["function"]["name"] == "get_positions"
    assert json.loads(call["function"]["arguments"]) == {"account_id": "X"}


def test_openai_tool_results_become_separate_tool_messages():
    # The mirror image of the Anthropic case: one message per result.
    out = _to_openai_messages(
        "s",
        [
            Message(
                role="user",
                tool_results=(
                    ToolResult(call_id="t1", content="ok"),
                    ToolResult(call_id="t2", content="boom", is_error=True),
                ),
            )
        ],
    )
    assert out[1:] == [
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        {"role": "tool", "tool_call_id": "t2", "content": "boom"},
    ]
