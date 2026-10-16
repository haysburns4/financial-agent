"""Pipelines must not hold SQLite's write lock while they wait on E-Trade."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import create_async_engine

from src.db import make_engine
from src.models import Base, PipelineRun, Position, Signal
from src.pipelines import portfolio_pipeline as portfolio_module
from src.pipelines.portfolio_pipeline import PortfolioPipeline
from src.pipelines.price_pipeline import PricePipeline

NOW = datetime(2026, 10, 16, 17, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'agent.db'}"
    engine = make_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Another writer that gives up fast, like any request arriving mid-run.
    other = create_async_engine(url, connect_args={"timeout": 0.2})
    yield engine, other
    await other.dispose()
    await engine.dispose()


async def _write_elsewhere(other) -> None:
    async with other.begin() as conn:
        await conn.execute(insert(Signal).values(
            ticker="NUE", timestamp=NOW, signal_type="rsi_oversold", direction="bullish",
            confidence=0.7, reasoning="r",
        ))


class _Market:
    def __init__(self, other) -> None:
        self._other = other

    async def get_quotes(self, tickers: list[str]) -> list[dict]:
        await _write_elsewhere(self._other)
        return [{
            "ticker": tickers[0], "timestamp": NOW, "last": 100.0, "close": 99.0,
            "open": 99.5, "high": 100.5, "low": 99.0, "volume": 1000.0,
        }]


class _Accounts:
    def __init__(self, other, fail: bool = False) -> None:
        self._other = other
        self._fail = fail

    async def list_accounts(self) -> list[dict]:
        if self._fail:
            raise RuntimeError("E-Trade is down")
        await _write_elsewhere(self._other)
        return [{"account_id": "84719991", "account_id_key": "key-1"}]

    async def get_positions(self, account_id_key: str) -> list[dict]:
        await _write_elsewhere(self._other)
        return [{"ticker": "AAPL", "quantity": 10, "costBasis": 1000.0, "marketValue": 1500.0}]


@pytest.fixture
def authenticated(monkeypatch):
    async def yes() -> bool:
        return True

    monkeypatch.setattr(portfolio_module.etrade_auth, "is_authenticated", yes)


async def _runs(engine) -> list[tuple[str, str]]:
    async with engine.connect() as conn:
        return [(r.pipeline, r.status) for r in (await conn.execute(select(PipelineRun))).all()]


async def test_the_price_pipeline_lets_others_write_while_it_fetches(db):
    engine, other = db

    result = await PricePipeline(_Market(other), engine).run(["AAPL", "MSFT"])

    assert result.errors == [] and result.status == "ok"  # each mid-fetch write got through
    assert await _runs(engine) == [("price_pipeline", "ok")]


async def test_the_portfolio_pipeline_lets_others_write_while_it_fetches(db, authenticated):
    engine, other = db

    result = await PortfolioPipeline(_Accounts(other), engine).run()

    assert result.errors == [] and result.status == "ok"
    async with engine.connect() as conn:
        assert (await conn.execute(select(Position.ticker))).scalars().all() == ["AAPL"]
    assert await _runs(engine) == [("portfolio_pipeline", "ok")]


async def test_a_run_that_raises_is_not_left_running(db, authenticated):
    engine, other = db

    with pytest.raises(RuntimeError, match="E-Trade is down"):
        await PortfolioPipeline(_Accounts(other, fail=True), engine).run()

    assert await _runs(engine) == [("portfolio_pipeline", "failed")]
