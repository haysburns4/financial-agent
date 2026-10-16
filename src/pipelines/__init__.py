"""Shared pipeline helpers."""
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.models import PipelineRun, Position


async def monitored_tickers(engine: AsyncEngine) -> list[str]:
    """Return the union of WATCHLIST and currently-held position tickers."""
    async with engine.connect() as conn:
        rows = (await conn.execute(select(Position.ticker).distinct())).all()
    held = {r.ticker for r in rows}
    watchlist = {t.strip().upper() for t in settings.WATCHLIST}
    return sorted(watchlist | held)


async def start_run(engine: AsyncEngine, pipeline: str, started: datetime) -> int:
    """Record that `pipeline` is running; returns the row's id for `finish_run`."""
    async with engine.begin() as conn:
        result = await conn.execute(
            sqlite_insert(PipelineRun).values(pipeline=pipeline, started_at=started, status="running")
        )
    return result.inserted_primary_key[0]


async def finish_run(
    engine: AsyncEngine, run_id: int, completed: datetime, status: str, processed: int, errors: list[str]
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id)
            .values(
                completed_at=completed,
                status=status,
                tickers_processed=processed,
                errors="\n".join(errors) if errors else None,
            )
        )
