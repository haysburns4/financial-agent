"""Reading the position snapshot history: rows, coverage, and a summary."""
from collections import Counter
from datetime import date, datetime, timezone

from sqlalchemy import Row, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from src.market_calendar import NYSE_TZ, trading_days
from src.models import PositionSnapshot, as_utc

# Longer gaps are cut to this many dates, with `missing_dates_truncated` set.
MISSING_DATES_CAP = 50


def snapshot_dict(row: Row) -> dict:
    return {
        "snapshot_date": row.snapshot_date.isoformat(),
        "account_id": row.account_id,
        "ticker": row.ticker,
        "quantity": row.quantity,
        "cost_basis": row.cost_basis,
        "market_value": row.market_value,
        "pnl": row.pnl,
        "pnl_pct": row.pnl_pct,
        "captured_at": as_utc(row.captured_at).isoformat(),
        "capture_source": row.capture_source,
        "market_state": row.market_state,
        "positions_as_of": as_utc(row.positions_as_of).isoformat(),
    }


async def history_rows(
    conn: AsyncConnection,
    *,
    account_id: str | None = None,
    ticker: str | None = None,
    start: date | None = None,
    end: date | None = None,
    limit: int = 500,
) -> list[dict]:
    """Snapshot rows, newest trading day first, then by ticker and account."""
    stmt = select(PositionSnapshot)
    if account_id is not None:
        stmt = stmt.where(PositionSnapshot.account_id == account_id)
    if ticker is not None:
        stmt = stmt.where(PositionSnapshot.ticker == ticker)
    if start is not None:
        stmt = stmt.where(PositionSnapshot.snapshot_date >= start)
    if end is not None:
        stmt = stmt.where(PositionSnapshot.snapshot_date <= end)
    stmt = stmt.order_by(
        PositionSnapshot.snapshot_date.desc(), PositionSnapshot.ticker, PositionSnapshot.account_id
    ).limit(limit)
    return [snapshot_dict(r) for r in (await conn.execute(stmt)).all()]


async def coverage(conn: AsyncConnection) -> dict:
    """Which trading days between the first and last snapshot have one."""
    per_date = (
        await conn.execute(
            select(PositionSnapshot.snapshot_date, func.max(PositionSnapshot.market_state))
            .group_by(PositionSnapshot.snapshot_date)
            .order_by(PositionSnapshot.snapshot_date)
        )
    ).all()
    if not per_date:
        return {
            "earliest_snapshot": None,
            "latest_snapshot": None,
            "total_snapshot_dates": 0,
            "trading_days_in_range": 0,
            "coverage_pct": None,
            "missing_dates": [],
            "missing_dates_truncated": False,
            "by_market_state": {},
        }

    recorded = {d for d, _ in per_date}
    earliest, latest = per_date[0][0], per_date[-1][0]
    expected = trading_days(earliest, latest)
    missing = [d for d in expected if d not in recorded]
    return {
        "earliest_snapshot": earliest.isoformat(),
        "latest_snapshot": latest.isoformat(),
        "total_snapshot_dates": len(recorded),
        "trading_days_in_range": len(expected),
        "coverage_pct": round(len(recorded & set(expected)) / len(expected), 4) if expected else None,
        "missing_dates": [d.isoformat() for d in missing[:MISSING_DATES_CAP]],
        "missing_dates_truncated": len(missing) > MISSING_DATES_CAP,
        "by_market_state": dict(Counter(state for _, state in per_date)),
    }


async def history_summary(conn: AsyncConnection, now: datetime | None = None) -> dict:
    """The /health "portfolio_history" section."""
    latest, dates = (
        await conn.execute(
            select(func.max(PositionSnapshot.snapshot_date), func.count(distinct(PositionSnapshot.snapshot_date)))
        )
    ).one()
    today = (now or datetime.now(timezone.utc)).astimezone(NYSE_TZ).date()
    return {
        "latest_snapshot_date": latest.isoformat() if latest else None,
        "snapshot_dates_recorded": dates,
        "days_since_last_snapshot": (today - latest).days if latest else None,
    }

