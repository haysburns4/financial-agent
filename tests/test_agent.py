"""The agent callers, exercised end-to-end against FakeBackend."""
from src.agent.chat import AgentChat
from src.agent.synthesizer import SignalSynthesizer
from src.llm import MessageComplete, TextDelta

from tests.conftest import FakeBackend, text_reply


async def test_ask_returns_the_answer_and_context(engine, backend):
    chat = AgentChat(engine, backend)
    result = await chat.ask("What do I hold?")

    assert result["answer"] == "AAPL is your largest position."
    # Empty DB: nothing to reference, and no bars means freshness is unknown.
    assert result["context_used"]["positions_referenced"] == []
    assert result["context_used"]["data_freshness_minutes"] == -1


async def test_ask_sends_the_question_as_the_final_turn(engine, backend):
    await AgentChat(engine, backend).ask("Why is NVDA down?")

    call = backend.calls[0]
    assert call["messages"][-1].role == "user"
    assert call["messages"][-1].text == "Why is NVDA down?"
    assert call["max_tokens"] == AgentChat.MAX_TOKENS


async def test_ask_replays_history_as_neutral_turns(engine, backend):
    await AgentChat(engine, backend).ask(
        "And MSFT?",
        [{"role": "user", "content": "How is AAPL?"}, {"role": "assistant", "content": "Flat."}],
    )

    roles = [(m.role, m.text) for m in backend.calls[0]["messages"]]
    assert roles == [
        ("user", "How is AAPL?"),
        ("assistant", "Flat."),
        ("user", "And MSFT?"),
    ]


async def test_ask_trims_history_to_the_cap(engine, backend):
    history = [{"role": "user", "content": f"q{i}"} for i in range(50)]
    await AgentChat(engine, backend).ask("latest", history)

    # MAX_HISTORY_TURNS user+assistant turns, plus the new question.
    assert len(backend.calls[0]["messages"]) == AgentChat.MAX_HISTORY_TURNS * 2 + 1


async def test_ask_flags_stale_data_when_etrade_is_unauthenticated(engine, backend):
    await AgentChat(engine, backend).ask("anything")
    assert "not currently authenticated with E-Trade" in backend.calls[0]["system"]


async def test_ask_stream_yields_deltas_then_the_terminal_event(engine):
    chat = AgentChat(engine, FakeBackend(text_reply("streamed")))
    deltas = [d async for d in chat.ask_stream("go")]

    assert isinstance(deltas[0], TextDelta)
    assert isinstance(deltas[-1], MessageComplete)
    assert deltas[-1].message.text == "streamed"


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
