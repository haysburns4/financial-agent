"""Neutral-layer semantics: what `collect` guarantees to callers."""
import pytest

from src.llm import (
    LLMError,
    Message,
    MessageComplete,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    collect,
)

from tests.conftest import FakeBackend, text_reply


async def _drain(deltas):
    return await collect(FakeBackend(deltas).stream(system="s", messages=[]))


async def test_collect_reads_the_terminal_message():
    # Text deltas are for the UI; the finished text comes from MessageComplete.
    response = await _drain(text_reply("hello"))
    assert response.text == "hello"
    assert response.stop_reason == "end_turn"
    assert response.usage == Usage(input_tokens=10, output_tokens=5)


async def test_collect_rejects_a_stream_with_no_terminal_event():
    # A backend that ends without MessageComplete is broken, not partially ok.
    with pytest.raises(LLMError):
        await _drain([TextDelta("par"), TextDelta("tial")])


async def test_collect_captures_tool_calls_and_stop_reason():
    call = ToolCall(id="t1", name="get_positions", arguments={"account_id": "X"})
    response = await _drain(
        [
            ToolCallDelta(index=0, id="t1", name="get_positions"),
            ToolCallDelta(index=0, arguments_json='{"account_id"'),
            ToolCallDelta(index=0, arguments_json=': "X"}'),
            MessageComplete(
                message=Message(role="assistant", tool_calls=(call,)),
                stop_reason="tool_use",
                usage=Usage(),
            ),
        ]
    )
    assert response.stop_reason == "tool_use"
    # Parsed arguments come off the terminal event, not the raw fragments.
    assert response.tool_calls == (call,)
