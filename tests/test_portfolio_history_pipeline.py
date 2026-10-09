"""Daily position snapshots: capture, same-day replacement, market states,
failure handling, and the history/coverage endpoints."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import insert, select

from src.models import PipelineRun, Position, PositionSnapshot
from src.pipelines.portfolio_pipeline import PortfolioRunResult
from src.pipelines.portfolio_history_pipeline import PortfolioHistoryPipeline

ET = ZoneInfo("America/New_York")
# Wednesday 2026-10-07, 11:00 ET: mid-session.
MIDSESSION = datetime(2026, 10, 7, 11, 0, tzinfo=ET)


class FakeAuth:
    def __init__(self, authenticated: bool = True) -> None:
        self.authenticated = authenticated

    async def is_authenticated(self) -> bool:
        return self.authenticated


class FakePortfolio:
    """Stands in for PortfolioPipeline: marks positions fresh, or fails."""

    def __init__(self, engine, clock, status: str = "ok", boom: bool = False) -> None:
        self.engine, self.clock, self.status, self.boom = engine, clock, status, boom
        self.runs = 0

    async def run(self) -> PortfolioRunResult:
        self.runs += 1
        if self.boom:
            raise RuntimeError("E-Trade timed out")
        if self.status == "ok":
            async with self.engine.begin() as conn:
                await conn.execute(Position.__table__.update().values(last_updated=self.clock()))
        now = self.clock()
        return PortfolioRunResult(now, now, self.status, 0, 0)


def _clock(moment: datetime):
    return lambda: moment.astimezone(timezone.utc)


async def _positions(engine, rows, age: timedelta = timedelta(minutes=5), at: datetime = MIDSESSION):
    async with engine.begin() as conn:
        await conn.execute(insert(Position), [
            {"account_id": a, "ticker": t, "quantity": q, "cost_basis": c, "market_value": m,
             "last_updated": (at - age).astimezone(timezone.utc)}
            for a, t, q, c, m in rows
        ])


TWO_ACCOUNTS = [
    ("84719991", "AAPL", 10.0, 1000.0, 1500.0),
    ("84719991", "MSFT", 5.0, 2000.0, 1800.0),
    ("55520002", "NVDA", 2.0, 0.0, 400.0),  # zero cost basis: pnl_pct is None
]


async def _snapshots(engine) -> list:
    async with engine.connect() as conn:
        return (await conn.execute(
            select(PositionSnapshot).order_by(PositionSnapshot.account_id, PositionSnapshot.ticker)
        )).all()


async def _runs(engine) -> list:
    async with engine.connect() as conn:
        return (await conn.execute(select(PipelineRun).where(PipelineRun.pipeline == "portfolio_history_pipeline"))).all()


def _pipeline(engine, moment=MIDSESSION, *, authenticated=True, portfolio=None) -> PortfolioHistoryPipeline:
    return PortfolioHistoryPipeline(engine, portfolio=portfolio, auth=FakeAuth(authenticated), clock=_clock(moment))


# ---------- capture ----------


async def test_writes_one_row_per_position_across_accounts(engine):
    await _positions(engine, TWO_ACCOUNTS)

    result = await _pipeline(engine).run(source="startup")

    rows = await _snapshots(engine)
    assert [(r.account_id, r.ticker) for r in rows] == [
        ("55520002", "NVDA"), ("84719991", "AAPL"), ("84719991", "MSFT"),
    ]
    aapl = rows[1]
    assert (aapl.pnl, aapl.pnl_pct) == (500.0, 0.5)
    assert rows[0].pnl_pct is None
    assert {(r.snapshot_date, r.market_state, r.capture_source) for r in rows} == {
        (date(2026, 10, 7), "intraday", "startup")
    }
    assert result.rows_written == 3 and not result.was_overwrite and result.errors == []
    assert result.positions_age_minutes == 5 and not result.positions_refreshed
    assert result.accounts == [
        {"account_id": "55520002", "position_count": 1, "total_market_value": 400.0},
        {"account_id": "84719991", "position_count": 2, "total_market_value": 3300.0},
    ]
    assert [r.status for r in await _runs(engine)] == ["ok"]


async def test_rerunning_the_same_day_replaces_rather_than_duplicates(engine):
    await _positions(engine, TWO_ACCOUNTS)
    await _pipeline(engine).run()

    # By the close, MSFT is sold and AAPL has moved.
    async with engine.begin() as conn:
        await conn.execute(Position.__table__.delete().where(Position.ticker == "MSFT"))
        await conn.execute(Position.__table__.update().where(Position.ticker == "AAPL").values(market_value=1600.0))
    after_close = datetime(2026, 10, 7, 17, 0, tzinfo=ET)
    result = await _pipeline(engine, after_close).run(refresh_first=False)

    rows = await _snapshots(engine)
    assert [(r.ticker, r.market_value) for r in rows] == [("NVDA", 400.0), ("AAPL", 1600.0)]
    assert {r.market_state for r in rows} == {"after_close"}  # the last capture wins
    assert result.was_overwrite and result.rows_written == 2


@pytest.mark.parametrize(
    ("moment", "snapshot_date", "market_state"),
    [
        (datetime(2026, 10, 7, 11, 0, tzinfo=ET), date(2026, 10, 7), "intraday"),
        (datetime(2026, 10, 7, 19, 0, tzinfo=ET), date(2026, 10, 7), "after_close"),
        (datetime(2026, 10, 7, 8, 0, tzinfo=ET), date(2026, 10, 7), "pre_open"),
        (datetime(2026, 10, 10, 12, 0, tzinfo=ET), date(2026, 10, 9), "weekend"),  # Saturday -> Friday
        (datetime(2026, 11, 26, 12, 0, tzinfo=ET), date(2026, 11, 25), "holiday"),  # Thanksgiving
        (datetime(2026, 11, 27, 14, 0, tzinfo=ET), date(2026, 11, 27), "after_close"),  # 1pm early close
    ],
)
async def test_snapshot_date_and_market_state_follow_the_nyse_calendar(engine, moment, snapshot_date, market_state):
    await _positions(engine, TWO_ACCOUNTS[:1], at=moment)

    result = await _pipeline(engine, moment).run()

    assert (result.snapshot_date, result.market_state) == (snapshot_date, market_state)
    assert {(r.snapshot_date, r.market_state) for r in await _snapshots(engine)} == {(snapshot_date, market_state)}


# ---------- stale positions and failures ----------


async def test_empty_positions_without_auth_writes_nothing_and_says_so(engine):
    portfolio = FakePortfolio(engine, _clock(MIDSESSION))

    result = await _pipeline(engine, authenticated=False, portfolio=portfolio).run(source="startup")

    assert await _snapshots(engine) == []
    assert result.rows_written == 0 and portfolio.runs == 0
    assert any("not authenticated" in e for e in result.errors)
    assert [r.status for r in await _runs(engine)] == ["auth_required"]


async def test_stale_positions_without_auth_are_recorded_with_their_age(engine):
    await _positions(engine, TWO_ACCOUNTS, age=timedelta(days=3))

    result = await _pipeline(engine, authenticated=False, portfolio=FakePortfolio(engine, _clock(MIDSESSION))).run()

    rows = await _snapshots(engine)
    assert len(rows) == 3 and not result.positions_refreshed
    assert result.positions_age_minutes == 3 * 24 * 60
    # The row itself shows its values are three days old.
    assert rows[0].positions_as_of.replace(tzinfo=timezone.utc) == (MIDSESSION - timedelta(days=3)).astimezone(timezone.utc)
    assert [r.status for r in await _runs(engine)] == ["partial"]


async def test_stale_positions_are_refreshed_first_when_authenticated(engine):
    await _positions(engine, TWO_ACCOUNTS, age=timedelta(hours=2))
    portfolio = FakePortfolio(engine, _clock(MIDSESSION))

    result = await _pipeline(engine, portfolio=portfolio).run()

    assert portfolio.runs == 1 and result.positions_refreshed
    assert result.positions_age_minutes == 0 and result.errors == []


async def test_fresh_positions_are_not_refreshed(engine):
    await _positions(engine, TWO_ACCOUNTS, age=timedelta(minutes=10))
    portfolio = FakePortfolio(engine, _clock(MIDSESSION))

    await _pipeline(engine, portfolio=portfolio).run()

    assert portfolio.runs == 0


async def test_a_failing_refresh_never_raises_into_the_caller(engine):
    await _positions(engine, TWO_ACCOUNTS, age=timedelta(hours=2))
    portfolio = FakePortfolio(engine, _clock(MIDSESSION), boom=True)

    result = await _pipeline(engine, portfolio=portfolio).run()

    assert result.rows_written == 0
    assert any("E-Trade timed out" in e for e in result.errors)
    assert [r.status for r in await _runs(engine)] == ["failed"]


async def test_a_stale_capture_is_retaken_after_login_but_a_fresh_one_is_not(engine):
    await _positions(engine, TWO_ACCOUNTS, age=timedelta(days=3))
    pipeline = _pipeline(engine, authenticated=False, portfolio=FakePortfolio(engine, _clock(MIDSESSION)))
    assert await pipeline.needs_capture()  # nothing captured today

    await pipeline.run(source="startup")
    assert await pipeline.needs_capture()  # captured, but from stale positions

    pipeline._auth = FakeAuth(True)
    result = await pipeline.run(source="startup")
    assert result.positions_refreshed and result.was_overwrite
    assert not await pipeline.needs_capture()


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


async def _history(engine, entries):
    """entries: (snapshot_date, account_id, ticker, market_state)."""
    async with engine.begin() as conn:
        await conn.execute(insert(PositionSnapshot), [
            {"snapshot_date": d, "account_id": a, "ticker": t, "market_state": s,
             "captured_at": datetime(d.year, d.month, d.day, 21, tzinfo=timezone.utc),
             "capture_source": "startup", "quantity": 1.0, "cost_basis": 100.0,
             "market_value": 110.0, "pnl": 10.0, "pnl_pct": 0.1,
             "positions_as_of": datetime(d.year, d.month, d.day, 21, tzinfo=timezone.utc)}
            for d, a, t, s in entries
        ])


HISTORY = [
    (date(2026, 11, 25), "84719991", "AAPL", "after_close"),
    (date(2026, 11, 25), "55520002", "NVDA", "after_close"),
    (date(2026, 12, 1), "84719991", "AAPL", "intraday"),
    (date(2026, 12, 1), "84719991", "MSFT", "intraday"),
    (date(2026, 12, 1), "55520002", "NVDA", "intraday"),
]


async def test_history_groups_by_account_newest_first(engine, client):
    await _history(engine, HISTORY)

    body = (await client.get("/portfolio/history")).json()

    assert set(body) == {"84719991", "55520002"}
    assert [(r["snapshot_date"], r["ticker"]) for r in body["84719991"]] == [
        ("2026-12-01", "AAPL"), ("2026-12-01", "MSFT"), ("2026-11-25", "AAPL"),
    ]
    row = body["84719991"][0]
    assert row["market_state"] == "intraday" and row["capture_source"] == "startup"
    assert row["captured_at"].startswith("2026-12-01T21:00:00")


async def test_history_filters_by_account_ticker_and_dates(engine, client):
    await _history(engine, HISTORY)

    one_account = (await client.get("/portfolio/history", params={"account_id": "84719991"})).json()
    by_ticker = (await client.get("/portfolio/history", params={"account_id": "84719991", "ticker": "aapl"})).json()
    by_dates = (await client.get("/portfolio/history", params={"start": "2026-11-26", "end": "2026-12-31"})).json()

    assert isinstance(one_account, list) and len(one_account) == 3
    assert [r["snapshot_date"] for r in by_ticker] == ["2026-12-01", "2026-11-25"]
    assert {r["snapshot_date"] for rows in by_dates.values() for r in rows} == {"2026-12-01"}


async def test_coverage_counts_missing_trading_days_not_holidays(engine, client):
    await _history(engine, HISTORY)

    body = (await client.get("/portfolio/history/coverage")).json()

    # 11/25 through 12/1 trades on 11/25, 11/27, 11/30 and 12/1; Thanksgiving
    # (11/26) and the weekend are not gaps.
    assert body == {
        "earliest_snapshot": "2026-11-25",
        "latest_snapshot": "2026-12-01",
        "total_snapshot_dates": 2,
        "trading_days_in_range": 4,
        "coverage_pct": 0.5,
        "missing_dates": ["2026-11-27", "2026-11-30"],
        "missing_dates_truncated": False,
        "by_market_state": {"after_close": 1, "intraday": 1},
    }


async def test_coverage_caps_a_long_gap(engine, client):
    await _history(engine, [(date(2026, 1, 2), "84719991", "AAPL", "after_close"),
                            (date(2026, 12, 31), "84719991", "AAPL", "after_close")])

    body = (await client.get("/portfolio/history/coverage")).json()

    assert len(body["missing_dates"]) == 50 and body["missing_dates_truncated"]
    assert body["missing_dates"][0] == "2026-01-05"


async def test_coverage_with_no_history(client):
    body = (await client.get("/portfolio/history/coverage")).json()
    assert body["total_snapshot_dates"] == 0 and body["latest_snapshot"] is None


async def test_manual_snapshot_endpoint_runs_the_pipeline(engine, client, monkeypatch):
    from src import server

    await _positions(engine, TWO_ACCOUNTS)
    monkeypatch.setattr(server, "portfolio_history_pipeline", _pipeline(engine))

    body = (await client.post("/pipeline/portfolio_history/run", json={"refresh_first": False})).json()

    assert body["capture_source"] == "manual" and body["rows_written"] == 3
    assert body["snapshot_date"] == "2026-10-07" and body["market_state"] == "intraday"
