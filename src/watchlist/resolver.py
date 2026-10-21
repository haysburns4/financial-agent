"""Which tickers are monitored.

Monitored = active watchlist rows ∪ tickers held with a nonzero quantity,
computed at call time and never stored.
"""
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from src.models import Indicator, Position, WatchlistItem

Source = Literal["watchlist", "held", "both"]


@dataclass(frozen=True)
class MonitoredTicker:
    ticker: str
    source: Source
    note: str | None
    tags: list[str]
    # Accounts holding a nonzero quantity, sorted.
    accounts: list[str]
    # Entry/exit rules have the history they need.
    indicators_ready: bool


async def get_monitored(engine: AsyncEngine) -> list[MonitoredTicker]:
    async with engine.connect() as conn:
        return await monitored_on(conn)


async def monitored_on(conn: AsyncConnection) -> list[MonitoredTicker]:
    """`get_monitored` on an open connection, for callers already inside one."""
    watched = {
        r.ticker: r
        for r in (await conn.execute(select(WatchlistItem).where(WatchlistItem.removed_at.is_(None)))).all()
    }
    holders: dict[str, list[str]] = {}
    for ticker, account_id in (
        await conn.execute(
            select(Position.ticker, Position.account_id)
            .where(Position.quantity != 0)
            .order_by(Position.account_id)
        )
    ).all():
        holders.setdefault(ticker, []).append(account_id)
    held = set(holders)
    with_history = (
        set(
            (
                await conn.scalars(
                    select(distinct(Indicator.ticker)).where(
                        Indicator.ticker.in_(held), Indicator.rsi_14.is_not(None)
                    )
                )
            ).all()
        )
        if held
        else set()
    )

    out: list[MonitoredTicker] = []
    for ticker in sorted(watched.keys() | held):
        row = watched.get(ticker)
        is_held = ticker in held
        source: Source = "both" if row is not None and is_held else "watchlist" if row is not None else "held"
        backfilled = row is not None and row.backfill_status == "done"
        out.append(
            MonitoredTicker(
                ticker=ticker,
                source=source,
                note=row.note if row is not None else None,
                tags=list(row.tags or []) if row is not None else [],
                accounts=holders.get(ticker, []),
                indicators_ready=backfilled or (is_held and ticker in with_history),
            )
        )
    return out


def tickers(monitored: list[MonitoredTicker]) -> list[str]:
    return [m.ticker for m in monitored]
