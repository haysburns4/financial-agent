"""The agent's tool surface: dispatch, argument coercion, failure handling."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert

from src.agent import tools
from src.agent.tools import TOOLS, Scope, data_summary, run_tool
from src.llm import ToolCall
from src.models import Indicator, Position, PriceBar, Signal, WatchlistItem


async def _seed(engine):
    async with engine.begin() as conn:
        await conn.execute(
            insert(Position),
            [
                {"account_id": "A1", "ticker": "AAPL", "quantity": 10.0,
                 "cost_basis": 1000.0, "market_value": 1500.0,
                 "last_updated": datetime.now(timezone.utc)},
                {"account_id": "A2", "ticker": "MSFT", "quantity": 5.0,
                 "cost_basis": 2000.0, "market_value": 1800.0,
                 "last_updated": datetime.now(timezone.utc)},
            ],
        )
        await conn.execute(
            insert(Signal),
            [
                {"ticker": "AAPL", "timestamp": datetime.now(timezone.utc),
                 "signal_type": "golden_cross", "direction": "up", "confidence": 0.75,
                 "reasoning": "ema9 crossed ema21"},
                {"ticker": "MSFT", "timestamp": datetime.now(timezone.utc),
                 "signal_type": "death_cross", "direction": "down", "confidence": 0.75,
                 "reasoning": "ema9 fell below ema21"},
            ],
        )


async def _call(engine, name: str, arguments: dict | None = None):
    result = await run_tool(engine, ToolCall(id="c1", name=name, arguments=arguments or {}))
    return result, json.loads(result.content) if not result.is_error else None


async def test_get_positions_returns_every_account(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_positions")

    # Ordered by market value descending (MSFT 1800, AAPL 1500).
    assert [p["ticker"] for p in payload] == ["MSFT", "AAPL"]
    # P&L is computed from cost basis, not stored.
    by_ticker = {p["ticker"]: p for p in payload}
    assert by_ticker["AAPL"]["pnl"] == 500.0
    assert by_ticker["MSFT"]["pnl"] == -200.0


async def test_get_positions_filters_by_account(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_positions", {"account_id": " A2 "})

    assert [p["ticker"] for p in payload] == ["MSFT"]


async def test_get_portfolio_risk_splits_by_account(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_portfolio_risk")

    assert payload["combined"]["position_count"] == 2
    assert set(payload["by_account"]) == {"A1", "A2"}


async def test_get_signals_filters_by_category(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_signals", {"category": "exit"})

    assert [s["signal_type"] for s in payload["top"]] == ["death_cross"]
    assert payload["top"][0]["category"] == "exit"


async def test_lowercase_tickers_are_normalised(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_signals", {"ticker": "aapl"})

    assert [s["ticker"] for s in payload["top"]] == ["AAPL"]


def _signal(ticker, signal_type, confidence, ts):
    return {"ticker": ticker, "timestamp": ts, "signal_type": signal_type, "direction": None,
            "confidence": confidence, "reasoning": "r"}


async def test_overview_ranks_the_top_ten_and_counts_the_rest(engine):
    now = datetime.now(timezone.utc)
    rows = [_signal(f"T{i:02d}", "breakout", 0.30 + i / 100, now - timedelta(minutes=i)) for i in range(15)]
    rows.append(_signal("OLD", "breakout", 0.99, now - timedelta(days=9)))  # outside the window
    async with engine.begin() as conn:
        await conn.execute(insert(Signal), rows)

    _, payload = await _call(engine, "get_signals")

    assert payload["total_signals"] == 15
    assert payload["by_type"] == {"breakout": 15}
    assert [s["ticker"] for s in payload["top"]] == [f"T{i:02d}" for i in range(14, 4, -1)]


async def test_position_alerts_are_listed_apart_not_ranked(engine):
    now = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(Signal), [
            _signal("AXP", "stop_loss_warning", 0.75, now - timedelta(hours=5)),
            _signal("AXP", "stop_loss_warning", 0.75, now - timedelta(hours=1)),  # repeat: newest kept
            _signal("BAM", "stop_loss_warning", 0.75, now),
            _signal("NVDA", "golden_cross", 0.41, now),
        ])

    _, payload = await _call(engine, "get_signals")

    assert [s["ticker"] for s in payload["top"]] == ["NVDA"]
    assert [(s["ticker"], s["signal_type"]) for s in payload["alerts"]] == [
        ("AXP", "stop_loss_warning"), ("BAM", "stop_loss_warning"),
    ]
    assert payload["total_signals"] == 1


async def test_signals_are_dated_by_when_they_fired_not_when_recorded(engine):
    # A catch-up writes old signals now; the window follows the bar's time.
    now = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(Signal), [_signal("AAPL", "breakout", 0.4, now - timedelta(days=12))])

    _, payload = await _call(engine, "get_signals")
    assert payload["total_signals"] == 0
    _, wide = await _call(engine, "get_signals", {"days": 30})
    assert wide["total_signals"] == 1


async def test_price_history_requires_a_ticker(engine):
    result, _ = await _call(engine, "get_price_history", {})

    assert result.is_error
    assert "ticker is required" in result.content


@pytest.mark.parametrize(
    "arguments",
    [{"limit": 10_000}, {"limit": -5}, {"limit": "many"}, {"limit": True}],
)
async def test_price_history_clamps_untrusted_limits(engine, arguments):
    # Arguments come from the model, so an absurd or mistyped limit must not
    # reach the query.
    result, payload = await _call(engine, "get_price_history", {"ticker": "AAPL", **arguments})

    assert not result.is_error
    assert payload["bar_count"] == 0  # no bars seeded; the point is it did not raise


async def _bar(engine, ticker: str, ts: datetime, rsi: float | None) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(PriceBar), [{
            "ticker": ticker, "timestamp": ts, "open": 100.0, "high": 100.0, "low": 100.0,
            "close": 177.89990234375, "volume": 1000.0, "adjusted_close": 100.0,
        }])
        await conn.execute(insert(Indicator), [{"ticker": ticker, "timestamp": ts, "rsi_14": rsi}])


async def test_screen_indicators_ranks_the_latest_bar_of_each_ticker_by_rsi(engine):
    await _seed(engine)
    async with engine.begin() as conn:
        await conn.execute(insert(WatchlistItem), [
            {"ticker": t, "tags": [], "added_at": datetime.now(timezone.utc), "backfill_status": "done"}
            for t in ("NUE", "V")
        ])
    now = datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc)
    await _bar(engine, "AAPL", now - timedelta(days=1), 20.0)  # superseded by the newer bar
    await _bar(engine, "AAPL", now, 61.234567)
    await _bar(engine, "NUE", now, 28.5)
    await _bar(engine, "MSFT", now, None)

    _, payload = await _call(engine, "screen_indicators", {"account_id": "all"})

    assert [(r["ticker"], r["rsi_14"], r["held"]) for r in payload["tickers"]] == [
        ("NUE", 28.5, False), ("AAPL", 61.23, True), ("MSFT", None, True),
    ]
    assert payload["tickers"][0]["close"] == 177.9  # float noise rounded off
    assert payload["no_indicators"] == ["V"]


async def test_unknown_tool_is_reported_not_raised(engine):
    result, _ = await _call(engine, "get_everything")

    assert result.is_error
    assert "unknown tool" in result.content


async def test_tool_failure_is_reported_as_an_error_result(engine, monkeypatch):
    async def _boom(conn, args, highlighted):
        raise RuntimeError("table is on fire")

    monkeypatch.setitem(tools._HANDLERS, "get_positions", _boom)
    result, _ = await _call(engine, "get_positions")

    # The model gets to see the failure and recover rather than the run dying.
    assert result.is_error
    assert "table is on fire" in result.content
    assert result.call_id == "c1"


async def test_data_summary_on_an_empty_database(engine):
    async with engine.begin() as conn:
        summary = await data_summary(conn)

    assert summary == {
        "positions": 0,
        "accounts": 0,
        "signals_recent": 0,
        "data_freshness_minutes": -1,
    }


async def test_data_summary_counts_accounts_not_positions(engine):
    await _seed(engine)
    async with engine.begin() as conn:
        summary = await data_summary(conn)

    assert summary["positions"] == 2
    assert summary["accounts"] == 2
    assert summary["signals_recent"] == 2


def test_every_tool_has_a_handler():
    assert {t.name for t in TOOLS} == set(tools._HANDLERS)


# ---------- the highlighted account ----------


async def _call_highlighted(engine, name: str, arguments: dict | None = None, highlighted: str | None = "A1"):
    result = await run_tool(engine, ToolCall(id="c1", name=name, arguments=arguments or {}), Scope(highlighted))
    return result, json.loads(result.content) if not result.is_error else None


async def test_account_tools_cover_the_highlighted_account_by_default(engine):
    await _seed(engine)

    _, positions = await _call_highlighted(engine, "get_positions")
    _, risk = await _call_highlighted(engine, "get_portfolio_risk")

    assert [p["account_id"] for p in positions] == ["A1"]
    assert risk["combined"]["position_count"] == 1
    assert set(risk["by_account"]) == {"A1"}


async def test_all_widens_an_account_tool_to_every_account(engine):
    await _seed(engine)

    _, positions = await _call_highlighted(engine, "get_positions", {"account_id": "ALL"})
    _, risk = await _call_highlighted(engine, "get_portfolio_risk", {"account_id": "all"})

    assert {p["account_id"] for p in positions} == {"A1", "A2"}
    assert set(risk["by_account"]) == {"A1", "A2"}


async def test_naming_another_account_overrides_the_highlight(engine):
    await _seed(engine)
    _, positions = await _call_highlighted(engine, "get_positions", {"account_id": "A2"})
    assert [p["ticker"] for p in positions] == ["MSFT"]


async def _seed_numbered(engine):
    now = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(Position), [
            {"account_id": acct, "ticker": ticker, "quantity": 1.0, "cost_basis": 100.0,
             "market_value": 100.0, "last_updated": now}
            for acct, ticker in [("84719991", "AAPL"), ("55520001", "MSFT"), ("66620001", "NVDA")]
        ])


@pytest.mark.parametrize("named", ["9991", "••9991", "84719991"])
async def test_an_account_can_be_named_by_its_last_digits(engine, named):
    await _seed_numbered(engine)
    _, positions = await _call_highlighted(engine, "get_positions", {"account_id": named}, highlighted=None)
    assert [p["account_id"] for p in positions] == ["84719991"]


@pytest.mark.parametrize(
    ("named", "problem"),
    [("1234", "is not a known account"), ("0001", "matches more than one account")],
)
async def test_an_unresolvable_account_is_an_error_listing_masked_accounts(engine, named, problem):
    await _seed_numbered(engine)
    result, _ = await _call_highlighted(engine, "get_positions", {"account_id": named}, highlighted=None)

    assert result.is_error and problem in result.content
    assert "••9991" in result.content and "84719991" not in result.content


async def test_position_alerts_are_narrowed_to_the_highlighted_account_but_market_signals_are_not(engine):
    await _seed(engine)  # A1 holds AAPL, A2 holds MSFT; AAPL golden_cross, MSFT death_cross
    now = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(Signal), [
            _signal("AAPL", "stop_loss_warning", 0.75, now),
            _signal("MSFT", "stop_loss_warning", 0.75, now),
        ])

    _, highlighted = await _call_highlighted(engine, "get_signals")
    _, everything = await _call_highlighted(engine, "get_signals", {"account_id": "all"})

    assert [s["ticker"] for s in highlighted["alerts"]] == ["AAPL"]
    assert {s["ticker"] for s in highlighted["top"]} == {"AAPL", "MSFT"}
    assert [s["ticker"] for s in everything["alerts"]] == ["AAPL", "MSFT"]


async def test_data_summary_counts_the_highlighted_accounts_positions(engine):
    await _seed(engine)
    async with engine.begin() as conn:
        summary = await data_summary(conn, "A2")

    assert summary["positions"] == 1
    assert summary["accounts"] == 2


# ---------- the Watchlist tab ----------


async def test_on_the_watchlist_tab_signals_cover_only_watchlist_tickers(engine):
    await _seed(engine)  # A1 holds AAPL, A2 holds MSFT; AAPL golden_cross, MSFT death_cross
    now = datetime.now(timezone.utc)
    async with engine.begin() as conn:
        await conn.execute(insert(WatchlistItem), [
            {"ticker": t, "tags": [], "added_at": now, "backfill_status": "done"} for t in ("NVDA", "MSFT")
        ])
        await conn.execute(insert(Signal), [
            _signal("NVDA", "breakout", 0.6, now),
            _signal("AAPL", "stop_loss_warning", 0.75, now),
            _signal("MSFT", "stop_loss_warning", 0.75, now),
        ])

    result = await run_tool(engine, ToolCall(id="c1", name="get_signals", arguments={}), Scope(watchlist_only=True))
    payload = json.loads(result.content)

    assert {s["ticker"] for s in payload["top"]} == {"NVDA", "MSFT"}
    assert [s["ticker"] for s in payload["alerts"]] == ["MSFT"]


async def test_on_the_watchlist_tab_the_screen_skips_held_only_tickers(engine):
    await _seed(engine)
    async with engine.begin() as conn:
        await conn.execute(insert(WatchlistItem).values(
            ticker="NVDA", tags=[], added_at=datetime.now(timezone.utc), backfill_status="done",
        ))

    result = await run_tool(
        engine, ToolCall(id="c1", name="screen_indicators", arguments={}), Scope(watchlist_only=True)
    )

    assert json.loads(result.content)["no_indicators"] == ["NVDA"]
