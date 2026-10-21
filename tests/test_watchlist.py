"""The database-backed watchlist: resolver, service, and its consumers.

In-memory SQLite, no network: the E-Trade symbol lookup and the yfinance
backfill are injected fakes.
"""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError

from src.agent.memory import build_context
from src.agent.tools import TOOLS
from src.etrade.market import QUOTE_SYMBOL_LIMIT, SymbolStatus
from src.models import Indicator, Position, PriceBar, Signal, WatchlistEvent, WatchlistItem
from src.pipelines.backfill_pipeline import BackfillResult
from src.pipelines.price_pipeline import PricePipeline
from src.signals.engine import FilterStats, SignalEngine
from src.watchlist.resolver import MonitoredTicker, get_monitored
from src.watchlist.service import WatchlistError, WatchlistService

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc)


class FakeLookup:
    def __init__(self, status: SymbolStatus = "valid", unknown: frozenset[str] = frozenset()) -> None:
        self.status = status
        self.unknown = unknown
        self.calls: list[str] = []

    async def lookup_symbol(self, ticker: str) -> SymbolStatus:
        self.calls.append(ticker)
        return "unknown" if ticker in self.unknown else self.status


class FakeBackfiller:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    async def run(self, tickers: list[str]) -> BackfillResult:
        self.calls.append(tickers)
        if self.fail:
            return BackfillResult(errors=[f"{tickers[0]}: {self.fail}"])
        return BackfillResult(tickers_processed=1, daily_bars_added=500, intraday_bars_added=3000)


class FakeMarket:
    """Quotes every symbol asked for, except the ones in `dead`; a batch that
    contains `broken` comes back empty, as get_quotes does on a failed request."""

    def __init__(self, broken: str | None = None) -> None:
        self.broken = broken
        self.requests: list[list[str]] = []

    async def get_quotes(self, tickers: list[str]) -> list[dict]:
        self.requests.append(list(tickers))
        if self.broken in tickers:
            return []
        return [
            {"ticker": t, "timestamp": NOW, "last": 100.0, "close": 99.0, "open": 99.5,
             "high": 100.5, "low": 99.0, "volume": 1000.0}
            for t in tickers
        ]


# The `engine` fixture's StaticPool is one SQLite connection for everyone, so a
# background backfill committing mid-mutation would commit (or roll back) that
# mutation's transaction too. Tests drain() backfills before the next write;
# a real pool gives each its own connection.
def make_service(engine, lookup=None, backfiller=None, max_size=50) -> WatchlistService:
    return WatchlistService(engine, lookup or FakeLookup(), backfiller or FakeBackfiller(), max_size)


async def hold(
    engine, ticker: str, quantity: float = 10.0, cost: float = 1000.0, value: float = 1000.0,
    account_id: str = "84719991",
) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(Position).values(
            account_id=account_id, ticker=ticker, quantity=quantity,
            cost_basis=cost, market_value=value, last_updated=NOW,
        ))


async def events(engine, ticker: str | None = None) -> list[tuple[str, str, str]]:
    stmt = select(WatchlistEvent).order_by(WatchlistEvent.id)
    if ticker:
        stmt = stmt.where(WatchlistEvent.ticker == ticker)
    async with engine.connect() as conn:
        return [(r.ticker, r.event, r.source) for r in (await conn.execute(stmt)).all()]


def by_ticker(monitored: list[MonitoredTicker]) -> dict[str, MonitoredTicker]:
    return {m.ticker: m for m in monitored}


# ---------- resolver ----------


async def test_resolver_labels_watchlist_held_and_both(engine):
    service = make_service(engine)
    await service.add("aapl", note="core", tags=["mega"])
    await service.add("NVDA")
    await service.drain()
    await hold(engine, "NVDA")
    await hold(engine, "V")
    await hold(engine, "V", account_id="20034412")
    await hold(engine, "SOLD", quantity=0)  # a closed position is not monitored
    await hold(engine, "NVDA", quantity=0, account_id="20034412")

    monitored = by_ticker(await get_monitored(engine))

    assert {t: m.source for t, m in monitored.items()} == {"AAPL": "watchlist", "NVDA": "both", "V": "held"}
    assert {t: m.accounts for t, m in monitored.items()} == {
        "AAPL": [], "NVDA": ["84719991"], "V": ["20034412", "84719991"],
    }
    assert monitored["AAPL"].note == "core" and monitored["AAPL"].tags == ["mega"]
    assert monitored["V"].note is None and monitored["V"].tags == []


async def test_indicators_ready_needs_a_finished_backfill_or_held_history(engine):
    service = make_service(engine, backfiller=FakeBackfiller(fail="yfinance timed out"))
    await service.add("PEND")
    await service.drain()  # failed
    await hold(engine, "HELD")
    await hold(engine, "NEW")
    async with engine.begin() as conn:
        await conn.execute(insert(Indicator).values(ticker="HELD", timestamp=NOW, rsi_14=55.0))
        await conn.execute(insert(Indicator).values(ticker="NEW", timestamp=NOW, rsi_14=None))  # warming up

    ready = {t: m.indicators_ready for t, m in by_ticker(await get_monitored(engine)).items()}

    assert ready == {"PEND": False, "HELD": True, "NEW": False}


# ---------- add / remove / restore ----------


async def test_add_remove_restore_keeps_note_and_tags(engine):
    service = make_service(engine)
    await service.add("msft", note="waiting on earnings", tags=["cloud", "cloud", " ai "])
    await service.drain()

    removed = await service.remove("MSFT", source="ui")
    assert removed.held is False
    assert "MSFT" not in by_ticker(await get_monitored(engine))

    restored = await service.add("MSFT", source="ui")
    await service.drain()

    assert restored.note == "waiting on earnings"
    assert restored.tags == ["cloud", "ai"]
    assert await events(engine) == [("MSFT", "added", "api"), ("MSFT", "removed", "ui"), ("MSFT", "restored", "ui")]
    async with engine.connect() as conn:
        assert await conn.scalar(select(func.count(WatchlistItem.id))) == 1  # the same row, restored


async def test_restore_takes_a_new_note_when_one_is_given(engine):
    service = make_service(engine)
    await service.add("MSFT", note="old", tags=["a"])
    await service.drain()
    await service.remove("MSFT")
    restored = await service.add("MSFT", note="new")
    await service.drain()
    assert (restored.note, restored.tags) == ("new", ["a"])


async def test_adding_an_active_ticker_changes_nothing(engine):
    lookup = FakeLookup()
    service = make_service(engine, lookup)
    first = await service.add("AAPL", note="keep")
    await service.drain()

    again = await service.add("aapl", note="ignored")

    assert again.note == "keep" and again.updated_at == (await service.get("AAPL")).updated_at
    assert await events(engine) == [("AAPL", "added", "api")]
    assert lookup.calls == ["AAPL"]
    assert first.ticker == again.ticker


async def test_remove_never_deletes_history(engine):
    service = make_service(engine)
    await service.add("AAPL")
    await service.drain()
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar).values(
            ticker="AAPL", timestamp=NOW, open=1, high=1, low=1, close=1, volume=1, adjusted_close=1,
        ))
        await conn.execute(insert(Indicator).values(ticker="AAPL", timestamp=NOW, rsi_14=40.0))
        await conn.execute(insert(Signal).values(
            ticker="AAPL", timestamp=NOW, signal_type="golden_cross", direction="long", confidence=0.6, reasoning="r",
        ))

    await service.remove("AAPL")

    async with engine.connect() as conn:
        for model in (PriceBar, Indicator, Signal):
            assert await conn.scalar(select(func.count()).select_from(model)) == 1, model.__name__


async def test_removing_a_held_ticker_says_it_stays_monitored(engine):
    service = make_service(engine)
    await service.add("NVDA")
    await service.drain()
    await hold(engine, "NVDA")

    result = await service.remove("NVDA")

    assert result.held is True
    assert by_ticker(await get_monitored(engine))["NVDA"].source == "held"
    assert await service.get("NVDA") is None


async def test_removing_what_is_not_on_the_watchlist_is_not_found(engine):
    with pytest.raises(WatchlistError) as caught:
        await make_service(engine).remove("ZZZ")
    assert caught.value.code == "not_found"


# ---------- validation ----------


@pytest.mark.parametrize("ticker", ["", "TOOLONGTICKER", "BRK B", "AB1", "^GSPC"])
async def test_badly_formed_tickers_are_rejected_before_any_lookup(engine, ticker):
    lookup = FakeLookup()
    with pytest.raises(WatchlistError) as caught:
        await make_service(engine, lookup).add(ticker)
    assert caught.value.code == "invalid_ticker"
    assert lookup.calls == []


async def test_share_classes_are_accepted(engine):
    service = make_service(engine)
    assert (await service.add("brk.b")).ticker == "BRK.B"
    assert (await service.add("BF-B")).ticker == "BF-B"
    await service.drain()


async def test_the_cap_is_enforced(engine):
    service = make_service(engine, max_size=2)
    await service.add("AAPL")
    await service.add("MSFT")
    await service.drain()

    with pytest.raises(WatchlistError) as caught:
        await service.add("NVDA")

    assert caught.value.code == "at_capacity"
    assert "2 of 2" in str(caught.value)
    # Making room lets it in.
    await service.remove("AAPL")
    await service.add("NVDA")
    await service.drain()


async def test_an_unknown_symbol_is_rejected(engine):
    service = make_service(engine, FakeLookup(unknown=frozenset({"ZZZZ"})))
    with pytest.raises(WatchlistError) as caught:
        await service.add("ZZZZ")
    assert caught.value.code == "unknown_symbol"
    assert await service.list_active() == []
    assert await events(engine) == []


async def test_an_add_while_e_trade_is_unavailable_is_accepted_unvalidated(engine):
    service = make_service(engine, FakeLookup(status="unavailable"))
    entry = await service.add("AAPL")
    await service.drain()
    assert entry.validated is False

    # The next successful price run confirms it.
    await PricePipeline(FakeMarket(), engine).run()
    assert (await service.get("AAPL")).validated is True


# ---------- backfill on add ----------


async def test_a_fresh_install_starts_with_an_empty_watchlist(engine):
    # Nothing is imported from settings: the user builds the list from scratch.
    assert await make_service(engine).list_active() == []
    assert await get_monitored(engine) == []


async def test_backfills_a_restart_interrupted_are_resumed(engine):
    async with engine.begin() as conn:
        await conn.execute(insert(WatchlistItem), [
            {"ticker": t, "tags": [], "added_at": NOW, "backfill_status": status}
            for t, status in (("AAPL", "pending"), ("MSFT", "running"), ("NVDA", "done"), ("V", "failed"))
        ])
    backfiller = FakeBackfiller()
    service = make_service(engine, backfiller=backfiller)

    assert sorted(await service.resume_backfills()) == ["AAPL", "MSFT"]
    await service.drain()
    assert sorted(backfiller.calls) == [["AAPL"], ["MSFT"]]


async def test_add_schedules_a_backfill_for_that_ticker_only(engine):
    backfiller = FakeBackfiller()
    service = make_service(engine, backfiller=backfiller)

    entry = await service.add("AAPL")
    assert entry.backfill_status == "pending"  # returned before the backfill ran
    await service.drain()

    assert backfiller.calls == [["AAPL"]]
    done = await service.get("AAPL")
    assert done.backfill_status == "done" and done.backfill_done_at is not None


async def test_a_failed_backfill_records_the_error_and_can_be_retried(engine):
    backfiller = FakeBackfiller(fail="yfinance returned nothing")
    service = make_service(engine, backfiller=backfiller)
    await service.add("AAPL")
    await service.drain()
    failed = await service.get("AAPL")
    assert failed.backfill_status == "failed"
    assert "yfinance returned nothing" in failed.backfill_error

    backfiller.fail = None
    await service.retry_backfill("AAPL")
    await service.drain()

    assert (await service.get("AAPL")).backfill_status == "done"
    assert (await service.get("AAPL")).backfill_error is None
    with pytest.raises(WatchlistError) as caught:
        await service.retry_backfill("AAPL")
    assert caught.value.code == "not_failed"


# ---------- events ----------


async def test_every_mutation_writes_an_event_with_old_and_new_values(engine):
    service = make_service(engine)
    await service.add("AAPL", note="a", tags=["x"], source="ui")
    await service.drain()
    await service.update("AAPL", note="b", source="ui")
    await service.update("AAPL", tags=["x", "y"], source="ui")
    await service.update("AAPL", note="b", tags=["x", "y"], source="ui")  # no change, no event
    await service.remove("AAPL", source="ui")
    await service.add("AAPL", source="ui")
    await service.drain()

    async with engine.connect() as conn:
        rows = (await conn.execute(select(WatchlistEvent).order_by(WatchlistEvent.id))).all()
    assert [r.event for r in rows] == ["added", "note_changed", "tags_changed", "removed", "restored"]
    assert {r.source for r in rows} == {"ui"}
    assert rows[1].detail == {"old_note": "a", "new_note": "b"}
    assert rows[2].detail == {"old_tags": ["x"], "new_tags": ["x", "y"]}


async def test_a_change_and_its_event_commit_together(engine):
    service = make_service(engine)
    # An event the schema rejects (bad source) must take the change down with it.
    with pytest.raises(IntegrityError):
        await service.add("AAPL", source="cron")
    assert await service.list_active() == []

    await service.add("AAPL", note="kept")
    await service.drain()
    with pytest.raises(IntegrityError):
        await service.update("AAPL", note="lost", source="cron")
    with pytest.raises(IntegrityError):
        await service.remove("AAPL", source="cron")

    entry = await service.get("AAPL")
    assert entry is not None and entry.note == "kept"
    assert await events(engine) == [("AAPL", "added", "api")]


# ---------- consumers ----------


async def test_the_price_pipeline_picks_up_a_ticker_added_after_startup(engine):
    service = make_service(engine)
    await service.add("AAPL")
    await service.drain()
    market = FakeMarket()
    pipeline = PricePipeline(market, engine)  # built once, as at startup

    await pipeline.run()
    await service.add("NVDA")
    await service.drain()
    await pipeline.run()

    assert market.requests == [["AAPL"], ["AAPL", "NVDA"]]


async def test_quotes_are_batched_and_a_failed_batch_drops_only_its_own(engine):
    tickers = [f"T{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(QUOTE_SYMBOL_LIMIT + 5)]
    market = FakeMarket(broken=tickers[-1])  # the second, short batch fails
    pipeline = PricePipeline(market, engine)

    result = await pipeline.run(tickers)

    assert [len(r) for r in market.requests] == [QUOTE_SYMBOL_LIMIT, 5]
    assert result.tickers_processed == QUOTE_SYMBOL_LIMIT
    assert result.status == "partial"
    assert sorted(e.split(":")[0] for e in result.errors) == tickers[QUOTE_SYMBOL_LIMIT:]
    # The failures count against each ticker's circuit breaker, as before batching.
    for _ in range(PricePipeline.FAILURE_THRESHOLD - 1):
        await pipeline.run(tickers)
    market.requests.clear()
    result = await pipeline.run(tickers)
    assert market.requests == [tickers[:QUOTE_SYMBOL_LIMIT]]
    assert result.tickers_skipped == 5


async def _crossing_bars(engine, ticker: str) -> None:
    """Two 5-minute bars whose EMAs cross upward: a golden_cross candidate."""
    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=10)
    if start.hour == 0 and start.minute < 10:
        start += timedelta(minutes=30)  # keep clear of the midnight daily-bar slot
    async with engine.begin() as conn:
        for i, (e9, e21) in enumerate([(99.0, 100.0), (101.0, 100.0)]):
            ts = start + timedelta(minutes=5 * i)
            await conn.execute(insert(PriceBar).values(
                ticker=ticker, timestamp=ts, open=100, high=101, low=99, close=100, volume=1000, adjusted_close=100,
            ))
            await conn.execute(insert(Indicator).values(ticker=ticker, timestamp=ts, ema_9=e9, ema_21=e21))


async def test_signal_engine_skips_entry_exit_until_indicators_are_ready_but_not_risk(engine):
    backfiller = FakeBackfiller(fail="still loading")
    service = make_service(engine, backfiller=backfiller)
    await service.add("NVDA")
    await service.drain()
    await _crossing_bars(engine, "NVDA")
    await hold(engine, "NVDA", cost=1000.0, value=900.0)  # -10%: stop-loss territory
    # Held, but its only indicator rows are the two above: rsi_14 is null, so not ready.
    assert by_ticker(await get_monitored(engine))["NVDA"].indicators_ready is False

    fired = {s["signal_type"] for s in await SignalEngine(engine, FilterStats()).run_all()}

    assert "golden_cross" not in fired
    assert {"stop_loss_warning", "concentration_risk"} <= fired

    # Once the backfill is done, entry rules run too.
    backfiller.fail = None
    await service.retry_backfill("NVDA")
    await service.drain()
    assert by_ticker(await get_monitored(engine))["NVDA"].indicators_ready is True
    fired = {s["signal_type"] for s in await SignalEngine(engine, FilterStats()).run_all()}
    assert "golden_cross" in fired


# ---------- agent context ----------


async def test_build_context_carries_notes_for_chat_and_only_scoped_ones_for_synthesis(engine):
    service = make_service(engine)
    await service.add("NVDA", note="waiting on earnings", tags=["semis"])
    await service.add("AAPL", note="trim above 250")
    await service.drain()
    await hold(engine, "V")

    chat = (await build_context(engine, "chat", 10_000)).render()
    synthesis = (await build_context(engine, "synthesis", 10_000, scope_tickers=["NVDA"])).render()

    assert "NVDA (on watchlist; tags: semis) — user's note: waiting on earnings" in chat
    assert "trim above 250" in chat and "- V (held; history still loading)" in chat
    assert "waiting on earnings" in synthesis
    assert "trim above 250" not in synthesis and "- V" not in synthesis


# ---------- the agent cannot change the watchlist ----------

_WRITE_ROUTE = re.compile(r"/watchlist", re.IGNORECASE)
_TOOL_HOOKS = ("useFrontendTool", "useCopilotAction", "useHumanInTheLoop", "defineTool", "frontendTools")


def test_no_llm_tool_or_copilotkit_action_can_write_the_watchlist():
    # Backend tools: none names or describes the watchlist as something it changes,
    # and nothing the agent runs imports the mutation service.
    for tool in TOOLS:
        assert "watchlist" not in tool.name.lower()
        assert not re.search(r"\b(add|remove|delete|edit|update)\b[^.]*watchlist", tool.description, re.IGNORECASE)
    agent_side = [*sorted((ROOT / "src" / "agent").glob("*.py")), ROOT / "src" / "agui.py", *sorted((ROOT / "src" / "llm").glob("*.py"))]
    for path in agent_side:
        text = path.read_text()
        assert "src.watchlist.service" not in text, path
        assert "WatchlistService" not in text and "watchlist_service" not in text, path
        assert not _WRITE_ROUTE.search(text), path

    # Frontend: a file that registers a CopilotKit tool or action never touches
    # the watchlist, and the CopilotKit route registers no tools at all.
    for path in sorted((ROOT / "web" / "app").rglob("*.ts*")):
        text = path.read_text()
        if any(hook in text for hook in _TOOL_HOOKS):
            assert "watchlist" not in text.lower(), path
    route = (ROOT / "web" / "app" / "api" / "copilotkit" / "[[...rest]]" / "route.ts").read_text()
    assert "watchlist" not in route.lower()
