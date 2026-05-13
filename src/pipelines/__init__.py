"""Shared pipeline helpers."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.models import Position


async def monitored_tickers(engine: AsyncEngine) -> list[str]:
    """Return the union of WATCHLIST and currently-held position tickers.

    This is the canonical "what to monitor" set. The watchlist is for
    securities the user wants to track without owning; positions are
    everything they actually hold across all E-Trade accounts. The union
    is what the price pipeline ingests and the signal engine evaluates.
    When a position is closed (no longer returned by E-Trade), the
    portfolio pipeline prunes it from `positions`, and it naturally drops
    out of this set on the next call.
    """
    async with engine.connect() as conn:
        rows = (await conn.execute(select(Position.ticker).distinct())).all()
    held = {r.ticker for r in rows}
    watchlist = {t.strip().upper() for t in settings.WATCHLIST}
    return sorted(watchlist | held)
