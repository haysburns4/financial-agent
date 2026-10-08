"""The agent callers, exercised end-to-end against FakeBackend."""
import json
from datetime import datetime, timezone

from sqlalchemy import insert

from src.agent.chat import AgentChat
from src.agent.synthesizer import SignalSynthesizer
from src.llm import Message, MessageComplete, TextDelta, ToolResult, Usage
from src.models import Position

from tests.conftest import FakeBackend, text_reply, tool_reply


async def _seed_position(engine):
    async with engine.begin() as conn:
        await conn.execute(
            insert(Position),
            [{"account_id": "A1", "ticker": "AAPL", "quantity": 10.0,
              "cost_basis": 1000.0, "market_value": 1500.0,
              "last_updated": datetime.now(timezone.utc)}],
        )


async def test_ask_returns_the_answer_and_context(engine, backend):
    result = await AgentChat(engine, backend).ask("What do I hold?")

    assert result["answer"] == "AAPL is your largest position."
    assert result["context_used"]["tools_called"] == []
    # Empty DB: no bars, so freshness is unknown rather than zero.
    assert result["context_used"]["data_freshness_minutes"] == -1


async def test_ask_sends_the_question_as_the_final_turn(engine, backend):
    await AgentChat(engine, backend).ask("Why is NVDA down?")

    call = backend.calls[0]
    assert call["messages"][-1] == Message(role="user", text="Why is NVDA down?")
    assert call["max_tokens"] == AgentChat.MAX_TOKENS


async def test_an_answer_cut_off_by_the_output_limit_says_so(engine):
    # What gpt-5 did at a 2048-token budget: all reasoning, no visible text.
    silent = [
        MessageComplete(
            message=Message(role="assistant", text=""),
            stop_reason="max_tokens",
            usage=Usage(input_tokens=7000, output_tokens=AgentChat.MAX_TOKENS),
        )
    ]
    backend = FakeBackend([tool_reply("get_positions", {}), silent])

    result = await AgentChat(engine, backend).ask("Analyze my allocation")

    assert "cut off" in result["answer"]


async def test_hitting_the_tool_ceiling_says_so(engine):
    turns = [tool_reply("get_positions", {}, call_id=f"c{i}") for i in range(AgentChat.MAX_TOOL_ITERATIONS)]
    result = await AgentChat(engine, FakeBackend(turns)).ask("Loop forever")

    assert "Stopped after" in result["answer"]


async def test_ask_replays_history_as_neutral_turns(engine, backend):
    await AgentChat(engine, backend).ask(
        "And MSFT?",
        [{"role": "user", "content": "How is AAPL?"}, {"role": "assistant", "content": "Flat."}],
    )

    assert [(m.role, m.text) for m in backend.calls[0]["messages"]] == [
        ("user", "How is AAPL?"),
        ("assistant", "Flat."),
        ("user", "And MSFT?"),
    ]


async def test_ask_trims_history_to_the_cap(engine, backend):
    history = [{"role": "user", "content": f"q{i}"} for i in range(50)]
    await AgentChat(engine, backend).ask("latest", history)

    assert len(backend.calls[0]["messages"]) == AgentChat.MAX_HISTORY_TURNS * 2 + 1


async def test_ask_flags_stale_data_when_etrade_is_unauthenticated(engine, backend):
    await AgentChat(engine, backend).ask("anything")
    assert "not currently authenticated with E-Trade" in backend.calls[0]["system"]


# ---------- tool surface ----------


async def test_the_tool_surface_is_offered_to_the_model(engine, backend):
    await AgentChat(engine, backend).ask("anything")

    names = {t.name for t in backend.calls[0]["tools"]}
    assert {"get_positions", "get_portfolio_risk", "get_signals", "get_price_history"} == names


async def test_system_prompt_orients_without_dumping_the_data(engine, backend):
    await _seed_position(engine)
    await AgentChat(engine, backend).ask("what do I hold?")

    system = backend.calls[0]["system"]
    assert "1 positions across 1 account(s)" in system
    # The whole point: figures come from tools, not from the prompt.
    assert "1500" not in system
    assert "AAPL" not in system.split("Watchlist:")[1].split("\n")[1]


async def test_agent_runs_a_tool_then_answers_with_the_result(engine, backend):
    await _seed_position(engine)
    backend = FakeBackend(
        [tool_reply("get_positions", {}), text_reply("You hold 1 position.")]
    )
    result = await AgentChat(engine, backend).ask("What do I hold?")

    assert result["answer"] == "You hold 1 position."
    assert result["context_used"]["tools_called"] == [{"name": "get_positions", "arguments": {}}]
    assert len(backend.calls) == 2


async def test_tool_results_are_replayed_as_a_user_turn(engine):
    await _seed_position(engine)
    backend = FakeBackend([tool_reply("get_positions", {}), text_reply("done")])
    await AgentChat(engine, backend).ask("holdings?")

    # Second turn carries: question, the assistant's tool call, then the results.
    second = backend.calls[1]["messages"]
    assert second[-2].tool_calls[0].name == "get_positions"
    results = second[-1].tool_results
    assert len(results) == 1
    assert results[0].call_id == "c1"
    assert not results[0].is_error
    assert "AAPL" in results[0].content


async def test_tool_loop_stops_at_the_ceiling(engine):
    # A model that only ever asks for more tools must not loop forever.
    backend = FakeBackend([tool_reply("get_positions", {})] * AgentChat.MAX_TOOL_ITERATIONS)
    result = await AgentChat(engine, backend).ask("holdings?")

    assert len(backend.calls) == AgentChat.MAX_TOOL_ITERATIONS
    assert len(result["context_used"]["tools_called"]) == AgentChat.MAX_TOOL_ITERATIONS


async def test_ask_stream_yields_tool_results_between_turns(engine):
    backend = FakeBackend([tool_reply("get_positions", {}), text_reply("streamed")])
    chat = AgentChat(engine, backend)

    kinds = [type(item).__name__ async for item in chat.ask_stream("go")]
    assert kinds == [
        "ToolCallDelta", "ToolCallDelta", "MessageComplete",  # turn 1: ask for a tool
        "ToolResult",                                        # executed server-side
        "TextDelta", "MessageComplete",                      # turn 2: the answer
    ]


async def test_ask_stream_yields_deltas_then_the_terminal_event(engine):
    chat = AgentChat(engine, FakeBackend(text_reply("streamed")))
    deltas = [d async for d in chat.ask_stream("go")]

    assert isinstance(deltas[0], TextDelta)
    assert isinstance(deltas[-1], MessageComplete)
    assert deltas[-1].message.text == "streamed"


# ---------- synthesizer ----------


async def test_synthesize_returns_the_briefing():
    backend = FakeBackend(text_reply("Two entries fired; both worth watching."))
    narrative = await SignalSynthesizer(backend).synthesize(
        [{"ticker": "AAPL", "signal_type": "breakout", "direction": "up", "confidence": 0.7,
          "reasoning": "closed above range", "timestamp": None}],
        {"positions": [], "bars_by_ticker": {}},
    )
    assert narrative == "Two entries fired; both worth watching."


async def test_synthesize_short_circuits_on_no_signals():
    backend = FakeBackend(text_reply("unused"))
    assert await SignalSynthesizer(backend).synthesize([], {}) == "No new signals."
    assert backend.calls == []  # no provider call, no spend


# ---------- the highlighted account ----------


async def _seed_two_accounts(engine):
    async with engine.begin() as conn:
        await conn.execute(
            insert(Position),
            [{"account_id": acct, "ticker": ticker, "quantity": 1.0, "cost_basis": 100.0,
              "market_value": 100.0, "last_updated": datetime.now(timezone.utc)}
             for acct, ticker in [("84719991", "AAPL"), ("55520002", "MSFT"), ("55520002", "NVDA")]],
        )


async def test_the_prompt_names_the_highlighted_account_masked(engine, backend):
    await _seed_two_accounts(engine)
    result = await AgentChat(engine, backend).ask("What do I hold?", account_id="84719991")

    system = backend.calls[0]["system"]
    assert "highlighted account ••9991" in system
    assert "1 positions in the highlighted account ••9991 (2 account(s) stored)" in system
    assert "84719991" not in system
    assert result["context_used"]["highlighted_account"] == "84719991"


async def test_tools_run_against_the_highlighted_account(engine):
    await _seed_two_accounts(engine)
    backend = FakeBackend([tool_reply("get_positions", {}), text_reply("Two positions.")])

    items = [i async for i in AgentChat(engine, backend).ask_stream("What do I hold?", account_id="55520002")]

    result = next(i for i in items if isinstance(i, ToolResult))
    assert {p["ticker"] for p in json.loads(result.content)} == {"MSFT", "NVDA"}


async def test_an_unknown_highlight_falls_back_to_every_account(engine, backend):
    await _seed_two_accounts(engine)
    result = await AgentChat(engine, backend).ask("What do I hold?", account_id="00000000")

    system = backend.calls[0]["system"]
    assert "highlighted account" not in system
    assert "3 positions across 2 account(s)" in system
    assert result["context_used"]["highlighted_account"] is None


async def test_no_highlight_analyzes_every_account(engine, backend):
    await _seed_two_accounts(engine)
    await AgentChat(engine, backend).ask("What do I hold?")
    assert "highlighted" not in backend.calls[0]["system"]
