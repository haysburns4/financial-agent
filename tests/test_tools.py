"""The agent's tool surface: dispatch, argument coercion, failure handling."""
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert

from src.agent import tools
from src.agent.tools import TOOLS, data_summary, run_tool
from src.llm import ToolCall
from src.models import Position, Signal


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
                 "reasoning": "ema9 crossed ema21", "delivered": False},
                {"ticker": "MSFT", "timestamp": datetime.now(timezone.utc),
                 "signal_type": "death_cross", "direction": "down", "confidence": 0.75,
                 "reasoning": "ema9 fell below ema21", "delivered": False},
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

    assert [s["signal_type"] for s in payload] == ["death_cross"]
    assert payload[0]["category"] == "exit"


async def test_lowercase_tickers_are_normalised(engine):
    await _seed(engine)
    _, payload = await _call(engine, "get_signals", {"ticker": "aapl"})

    assert [s["ticker"] for s in payload] == ["AAPL"]


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


async def test_unknown_tool_is_reported_not_raised(engine):
    result, _ = await _call(engine, "get_everything")

    assert result.is_error
    assert "unknown tool" in result.content


async def test_tool_failure_is_reported_as_an_error_result(engine, monkeypatch):
    async def _boom(conn, args):
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
        "signals_24h": 0,
        "data_freshness_minutes": -1,
    }


async def test_data_summary_counts_accounts_not_positions(engine):
    await _seed(engine)
    async with engine.begin() as conn:
        summary = await data_summary(conn)

    assert summary["positions"] == 2
    assert summary["accounts"] == 2
    assert summary["signals_24h"] == 2


def test_every_tool_has_a_handler():
    assert {t.name for t in TOOLS} == set(tools._HANDLERS)
