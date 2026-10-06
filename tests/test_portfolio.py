"""The dashboard payload behind the web UI's portfolio panel."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select

from src.models import Position
from src.portfolio import dashboard

NOW = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)


async def _rows(engine, positions: list[dict]):
    async with engine.begin() as conn:
        if positions:
            await conn.execute(insert(Position), positions)
        return (await conn.execute(select(Position))).all()


def _pos(account: str, ticker: str, cost: float, value: float, age_min: int = 0) -> dict:
    return {
        "account_id": account, "ticker": ticker, "quantity": 1.0,
        "cost_basis": cost, "market_value": value,
        "last_updated": NOW - timedelta(minutes=age_min),
    }


async def test_totals_and_accounts(engine):
    rows = await _rows(
        engine,
        [
            _pos("B2", "MSFT", 2000.0, 1800.0),
            _pos("A1", "AAPL", 1000.0, 1500.0),
            _pos("A1", "NVDA", 500.0, 500.0),
        ],
    )
    data = dashboard(rows)

    assert data["totals"] == {
        "market_value": 3800.0,
        "cost_basis": 3500.0,
        "pnl": 300.0,
        "pnl_pct": pytest.approx(300 / 3500),
        "position_count": 3,
        "account_count": 2,
    }
    # Accounts sorted by id; positions within each by market value, largest first.
    assert [a["account_id"] for a in data["accounts"]] == ["A1", "B2"]
    a1 = data["accounts"][0]
    assert [p["ticker"] for p in a1["positions"]] == ["AAPL", "NVDA"]
    assert a1["pnl"] == 500.0


async def test_last_updated_is_the_newest_position(engine):
    rows = await _rows(engine, [_pos("A1", "AAPL", 1, 1, age_min=30), _pos("A1", "MSFT", 1, 1)])
    assert dashboard(rows)["last_updated"].replace(tzinfo=timezone.utc) == NOW


async def test_empty_portfolio(engine):
    data = dashboard(await _rows(engine, []))

    assert data["accounts"] == []
    assert data["last_updated"] is None
    assert data["totals"]["pnl_pct"] is None
    assert data["totals"]["account_count"] == 0


async def test_zero_cost_basis_has_no_pnl_pct(engine):
    data = dashboard(await _rows(engine, [_pos("A1", "GIFT", 0.0, 100.0)]))
    assert data["accounts"][0]["pnl_pct"] is None
    assert data["accounts"][0]["positions"][0]["pnl_pct"] is None
