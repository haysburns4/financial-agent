"""The agent output archive and the memory assembler: what is recorded, how it
is indexed and served, and how background is chosen and fitted to a budget."""
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from loguru import logger
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from src.agent.archive import content_hash, record_output, refs_from_context
from src.agent.chat import AgentChat, conversation_key
from src.agent.memory import build_context, context_budget, estimate_tokens
from src.agent.synthesizer import SignalSynthesizer
from src.config import settings
from src.models import AgentOutput, AgentOutputRef, PositionSnapshot

from tests.conftest import FakeBackend, text_reply, tool_reply

SIGNAL = {"id": 7, "ticker": "NVDA", "signal_type": "breakout", "direction": "up",
          "confidence": 0.7, "reasoning": "closed above range", "timestamp": None}


@pytest.fixture
def logs():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(f"{m.record['level'].name} {m.record['message']}"))
    yield lines
    logger.remove(sink)


async def _outputs(engine) -> list:
    async with engine.connect() as conn:
        return (await conn.execute(select(AgentOutput).order_by(AgentOutput.id))).all()


async def _refs(engine, output_id: int) -> list[tuple[str, str]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            select(AgentOutputRef.ref_type, AgentOutputRef.ref_value)
            .where(AgentOutputRef.output_id == output_id).order_by(AgentOutputRef.id)
        )
        return [tuple(r) for r in rows.all()]


async def _narrative(engine, text: str, tickers: list[str], days_ago: float = 1) -> int:
    output_id = await record_output(engine, "synthesis", text, "anthropic", "claude-sonnet-5",
                                    context_ref={"tickers": tickers})
    async with engine.begin() as conn:
        await conn.execute(
            AgentOutput.__table__.update().where(AgentOutput.id == output_id)
            .values(timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago))
        )
    return output_id


# ---------- archive: write path ----------


async def test_synthesis_is_archived_with_its_provider_model_and_refs(engine):
    backend = FakeBackend(text_reply("NVDA broke out; worth watching."), model="claude-sonnet-5")
    context = {"positions": [{"account_id": "84719991", "ticker": "NVDA", "quantity": 2.0,
                              "cost_basis": 100.0, "market_value": 150.0}], "bars_by_ticker": {}}

    narrative = await SignalSynthesizer(backend, engine).synthesize([SIGNAL], context)

    [row] = await _outputs(engine)
    assert row.content == narrative == "NVDA broke out; worth watching."
    assert (row.output_type, row.role, row.provider, row.model) == ("synthesis", None, "fake", "claude-sonnet-5")
    assert row.context_ref["signal_ids"] == [7] and row.latency_ms is not None and not row.delivered
    assert await _refs(engine, row.id) == [("signal_id", "7"), ("ticker", "NVDA"), ("account_id", "84719991")]


async def test_a_chat_turn_is_archived_as_two_rows_sharing_a_conversation(engine):
    backend = FakeBackend([tool_reply("get_price_history", {"ticker": "nvda"}), text_reply("NVDA is trending up.")])

    result = await AgentChat(engine, backend).ask("How is NVDA doing?")

    user, assistant = await _outputs(engine)
    assert (user.role, user.content) == ("user", "How is NVDA doing?")
    assert (assistant.role, assistant.content) == ("assistant", "NVDA is trending up.")
    assert user.conversation_id == assistant.conversation_id == result["conversation_id"]
    # The assistant row's context_ref is the response's context_used.
    assert assistant.context_ref == result["context_used"]
    assert ("ticker", "NVDA") in await _refs(engine, assistant.id)


async def test_a_supplied_conversation_id_threads_the_turn(engine, backend):
    result = await AgentChat(engine, backend).ask("What do I hold?", conversation_id="thread-1")

    assert result["conversation_id"] == "thread-1"
    assert {r.conversation_id for r in await _outputs(engine)} == {"thread-1"}


def test_conversation_ids_are_generated_or_made_to_fit():
    assert len(conversation_key(None)) == 36  # uuid4
    assert conversation_key(" t1 ") == "t1"
    long_id = conversation_key("x" * 100)
    assert len(long_id) == 40 and long_id == conversation_key("x" * 100)


def test_content_hash_is_stable_sha256():
    assert content_hash("same text") == content_hash("same text")
    assert content_hash("same text") != content_hash("same text.")
    assert len(content_hash("")) == 64


def test_refs_are_found_at_any_depth_normalised_and_deduplicated():
    context_ref = {
        "highlighted_account": "84719991",
        "tools_called": [
            {"name": "get_positions", "arguments": {"account_id": "all"}},  # "all" is not an account
            {"name": "get_signals", "arguments": {"ticker": "nvda"}},
            {"name": "get_price_history", "arguments": {"ticker": "NVDA", "limit": 5}},
        ],
        "signal_ids": [3, 4],
        "data_freshness_minutes": 5,
    }
    assert refs_from_context(context_ref) == [
        ("account_id", "84719991"), ("ticker", "NVDA"), ("signal_id", "3"), ("signal_id", "4"),
    ]


async def test_a_failed_archive_write_logs_and_returns_none(logs):
    no_tables = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)

    assert await record_output(no_tables, "chat", "hi", "fake", "fake-1", role="user") is None
    assert any(line.startswith("ERROR archive: could not record chat output") for line in logs)
    await no_tables.dispose()


async def test_a_broken_archive_does_not_fail_synthesis(logs):
    no_tables = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    backend = FakeBackend(text_reply("Briefing."))

    assert await SignalSynthesizer(backend, no_tables).synthesize([SIGNAL], {}) == "Briefing."
    assert any(line.startswith("ERROR") for line in logs)
    await no_tables.dispose()


# ---------- memory ----------


async def test_synthesis_memory_has_prior_briefings_on_its_tickers_and_no_conversation(engine):
    nvda = await _narrative(engine, "NVDA broke out yesterday.", ["NVDA"])
    await _narrative(engine, "AAPL slipped.", ["AAPL"])
    await _narrative(engine, "NVDA a quarter ago.", ["NVDA"], days_ago=90)  # outside the lookback
    await record_output(engine, "chat", "a chat turn", "fake", "fake-1", role="user", conversation_id="c1")

    memory = await build_context(engine, "synthesis", 10_000, scope_tickers=["NVDA"], conversation_id="c1")

    assert [s.label for s in memory.sections] == ["prior_narratives"]
    text = memory.render()
    assert "NVDA broke out yesterday." in text
    assert "AAPL slipped." not in text and "a quarter ago" not in text and "chat turn" not in text
    assert memory.output_ids == [nvda]


async def test_synthesis_memory_shows_the_latest_snapshot_of_its_tickers(engine):
    async with engine.begin() as conn:
        await conn.execute(insert(PositionSnapshot), [
            {"snapshot_date": d, "captured_at": datetime.now(timezone.utc), "capture_source": "startup",
             "market_state": "after_close", "account_id": "84719991", "ticker": t, "quantity": 2.0,
             "cost_basis": 100.0, "market_value": mv, "pnl": mv - 100.0, "pnl_pct": (mv - 100.0) / 100.0,
             "positions_as_of": datetime.now(timezone.utc)}
            for d, t, mv in [(date(2026, 10, 8), "NVDA", 150.0), (date(2026, 10, 9), "NVDA", 160.0),
                             (date(2026, 10, 9), "AAPL", 90.0)]
        ])

    memory = await build_context(engine, "synthesis", 10_000, scope_tickers=["NVDA"])

    portfolio = next(s for s in memory.sections if s.label == "portfolio").content
    assert portfolio == "NVDA in ••9991 on 2026-10-09 (after_close): 2 shares, $160, P&L +$60 (+60.0%)"


async def test_chat_memory_includes_earlier_turns_not_already_replayed(engine):
    for i, (role, text) in enumerate([("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]):
        await record_output(engine, "chat", text, "fake", "fake-1", role=role, conversation_id="c1")
        async with engine.begin() as conn:  # distinct, ordered timestamps
            await conn.execute(
                AgentOutput.__table__.update().where(AgentOutput.content == text)
                .values(timestamp=datetime(2026, 10, 9, 12, i, tzinfo=timezone.utc))
            )
    await record_output(engine, "chat", "other conversation", "fake", "fake-1", role="user", conversation_id="c2")

    everything = await build_context(engine, "chat", 10_000, conversation_id="c1")
    older = await build_context(engine, "chat", 10_000, conversation_id="c1", skip_recent_turns=2)

    conversation = next(s for s in everything.sections if s.label == "conversation").content
    assert [line.split(" (")[0] for line in conversation.split("\n\n")] == ["user", "assistant", "user", "assistant"]
    assert conversation.index("q1") < conversation.index("a2") and "other conversation" not in conversation
    older_text = next(s for s in older.sections if s.label == "conversation").content
    assert "q1" in older_text and "a1" in older_text and "q2" not in older_text


async def test_build_context_fits_its_budget_and_says_what_it_cut(engine):
    for i in range(5):
        await _narrative(engine, f"NVDA briefing {i}. " + "detail " * 2000, ["NVDA"], days_ago=i + 1)

    memory = await build_context(engine, "synthesis", 3_000, scope_tickers=["NVDA"])

    assert memory.tokens_estimated <= memory.budget_tokens == 3_000
    assert memory.tokens_estimated == estimate_tokens(memory.render())
    assert memory.truncated == ["prior_narratives"]
    assert "NVDA briefing 0." in memory.render()  # newest kept, trimmed to fit


async def test_a_zero_budget_yields_nothing_but_records_the_cut(engine):
    await _narrative(engine, "NVDA broke out.", ["NVDA"])

    memory = await build_context(engine, "synthesis", 0, scope_tickers=["NVDA"])

    assert memory.sections == [] and memory.tokens_estimated == 0
    assert memory.truncated == ["prior_narratives"]


async def test_chat_puts_memory_in_the_system_prompt_and_records_what_it_used(engine):
    briefing = await _narrative(engine, "Semis rallied on Friday.", ["NVDA"])
    backend = FakeBackend(text_reply("Noted."))

    result = await AgentChat(engine, backend).ask("Anything I should know?")

    assert "Semis rallied on Friday." in backend.calls[0]["system"]
    assert result["context_used"]["memory"]["output_ids"] == [briefing]


async def test_a_synthesis_sees_and_records_the_briefings_before_it(engine):
    earlier = await _narrative(engine, "NVDA broke out yesterday.", ["NVDA"])
    backend = FakeBackend(text_reply("NVDA's breakout held."))

    await SignalSynthesizer(backend, engine).synthesize([SIGNAL], {"positions": [], "bars_by_ticker": {}})

    assert "NVDA broke out yesterday." in backend.calls[0]["messages"][0].text
    latest = (await _outputs(engine))[-1]
    assert latest.context_ref["memory"]["output_ids"] == [earlier]


@pytest.mark.parametrize(
    ("overrides", "chat", "synthesis"),
    [
        ({"LLM_PROVIDER": "anthropic"}, 120_000, 120_000),
        ({"LLM_PROVIDER": "local"}, 24_000, 24_000),
        ({"LLM_PROVIDER": "anthropic", "LLM_PROVIDER_SYNTHESIZER": "local"}, 120_000, 24_000),
    ],
)
def test_the_budget_follows_the_provider_each_task_routes_to(overrides, chat, synthesis):
    config = settings.model_copy(update={
        "LLM_PROVIDER_CHAT": None, "LLM_PROVIDER_SYNTHESIZER": None,
        "LLM_CONTEXT_BUDGET": 120_000, "LOCAL_LLM_CONTEXT_BUDGET": 24_000, **overrides,
    })
    assert (context_budget("chat", config), context_budget("synthesis", config)) == (chat, synthesis)


# ---------- endpoints ----------


@pytest.fixture
async def client(engine):
    from src import server

    app = server.create_app()

    async def conn_dep():
        async with engine.begin() as conn:
            yield conn

    app.dependency_overrides[server._conn_dep] = conn_dep
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_history_filters_by_ticker_through_the_refs(engine, client):
    nvda = await _narrative(engine, "NVDA broke out.", ["NVDA"])
    await _narrative(engine, "AAPL slipped.", ["AAPL"])
    await record_output(engine, "chat", "What about NVDA?", "fake", "fake-1", role="user", conversation_id="c1")

    by_ticker = (await client.get("/agent/history", params={"ticker": "nvda"})).json()
    by_type = (await client.get("/agent/history", params={"type": "chat"})).json()
    by_conversation = (await client.get("/agent/history", params={"conversation_id": "c1"})).json()

    # The chat question names NVDA only in its text, which is not indexed.
    assert [o["id"] for o in by_ticker] == [nvda]
    assert [o["content"] for o in by_type] == [o["content"] for o in by_conversation] == ["What about NVDA?"]


async def test_history_is_newest_first_and_filters_by_date(engine, client):
    await _narrative(engine, "three days ago", ["NVDA"], days_ago=3)
    await _narrative(engine, "one day ago", ["NVDA"], days_ago=1)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

    everything = (await client.get("/agent/history")).json()
    recent = (await client.get("/agent/history", params={"start": yesterday})).json()

    assert [o["content"] for o in everything] == ["one day ago", "three days ago"]
    assert [o["content"] for o in recent] == ["one day ago"]


async def test_one_output_comes_with_its_context_and_refs(engine, client):
    output_id = await _narrative(engine, "NVDA broke out.", ["NVDA"])

    body = (await client.get(f"/agent/history/{output_id}")).json()

    assert body["context_ref"] == {"tickers": ["NVDA"]}
    assert body["refs"] == [{"ref_type": "ticker", "ref_value": "NVDA"}]
    assert (await client.get("/agent/history/999")).status_code == 404


async def test_undelivered_lists_syntheses_not_yet_delivered(engine, client):
    first = await _narrative(engine, "first", ["NVDA"], days_ago=2)
    second = await _narrative(engine, "second", ["NVDA"])
    async with engine.begin() as conn:
        await conn.execute(AgentOutput.__table__.update().where(AgentOutput.id == first).values(delivered=True))
    await record_output(engine, "chat", "not a synthesis", "fake", "fake-1", role="user")

    body = (await client.get("/agent/history/undelivered", params={"type": "synthesis"})).json()

    assert [o["id"] for o in body] == [second]


async def test_archive_summary_counts_for_health(engine):
    from src.agent.archive import archive_summary

    async with engine.connect() as conn:
        empty = await archive_summary(conn)
    await _narrative(engine, "briefing", ["NVDA"])
    await record_output(engine, "chat", "q", "fake", "fake-1", role="user", conversation_id="c1")
    await record_output(engine, "chat", "a", "fake", "fake-1", role="assistant", conversation_id="c1")
    async with engine.connect() as conn:
        summary = await archive_summary(conn)

    assert empty == {"total_outputs": 0, "syntheses": 0, "chat_turns": 0,
                     "undelivered_syntheses": 0, "latest_output_at": None}
    assert {k: v for k, v in summary.items() if k != "latest_output_at"} == {
        "total_outputs": 3, "syntheses": 1, "chat_turns": 2, "undelivered_syntheses": 1,
    }
    assert summary["latest_output_at"] is not None


def test_estimate_tokens_counts_each_digit_and_mark():
    # Qwen splits numbers digit by digit; at four characters a token, price
    # JSON was undercounted about threefold.
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("177.8999") == 8
